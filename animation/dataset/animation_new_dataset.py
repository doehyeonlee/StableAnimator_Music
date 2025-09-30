import os, io, csv, math, random
from importlib.metadata import files
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from einops import rearrange
import cv2
import warnings
import h5py


class LargeScaleMusicVideos(Dataset):
    def __init__(self, root_path, txt_path, width, height, n_sample_frames, sample_frame_rate, sample_margin=30,
                 crop_len=None, app=None, handler_ante=None, face_helper=None):
        self.root_path = root_path
        self.txt_path = txt_path
        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_frame_rate = sample_frame_rate
        self.sample_margin = sample_margin
        self.crop_len = crop_len

        self.video_files = self._read_txt_file_images()

        self.app = app
        self.handler_ante = handler_ante
        self.face_helper = face_helper

    def _read_txt_file_images(self):
        with open(self.txt_path, 'r') as file:
            lines = file.readlines()
            video_files = []
            for line in lines:
                video_file = line.strip()
                video_files.append(video_file)
        return video_files

    def __len__(self):
        return len(self.video_files)

    def frame_count(self, frames_path):
        files = os.listdir(frames_path)
        png_files = [file for file in files if file.endswith('.png') or file.endswith('.jpg')]
        png_files_count = len(png_files)
        return png_files_count

    def find_frames_list(self, frames_path):
        files = os.listdir(frames_path)
        image_files = [file for file in files if file.endswith('.png') or file.endswith('.jpg')]
        if image_files[0].startswith('frame_'):
            image_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        else:
            image_files.sort(key=lambda x: int(x.split('.')[0]))
        return image_files

    def get_face_masks(self, pil_img):
        print("We don't want to use this function anymore.")
        rgb_image = np.array(pil_img)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        image_info = self.app.get(bgr_image)
        mask = np.zeros((self.height, self.width), dtype=np.uint8)

        if len(image_info) > 0:
            for info in image_info:
                x_1 = info['bbox'][0]
                y_1 = info['bbox'][1]
                x_2 = info['bbox'][2]
                y_2 = info['bbox'][3]
                cv2.rectangle(mask, (int(x_1), int(y_1)), (int(x_2), int(y_2)), (255), thickness=cv2.FILLED)
            mask = mask.astype(np.float64) / 255.0
        else:
            self.face_helper.clean_all()
            with torch.no_grad():
                bboxes = self.face_helper.face_det.detect_faces(bgr_image, 0.97)
            if len(bboxes) > 0:
                for bbox in bboxes:
                    cv2.rectangle(mask, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255),
                                  thickness=cv2.FILLED)
                mask = mask.astype(np.float64) / 255.0
            else:
                mask = np.ones((self.height, self.width), dtype=np.uint8)
        return mask

    def resize_and_center_crop(self, img, target_width, target_height):
        width, height = img.size
        if width <= target_width and height <= target_height:
            return img.resize((target_width, target_height), Image.LANCZOS)
        scale = max(target_width / width, target_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Center crop
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        
        return img.crop((left, top, right, bottom))

    def __getitem__(self, idx):
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)
        base_path = '/root/dataset/rec/'
        video_base_path = self.video_files[idx]
        frames_path = osp.join(video_base_path, "images")
        poses_path = osp.join(video_base_path, "poses")
        #face_masks_path = osp.join(video_base_path, "faces")
        seg_masks_path = osp.join(video_base_path, "seg")
        video_length = self.frame_count(frames_path)
        frames_list = self.find_frames_list(frames_path)
        mus_path = video_base_path.split('/')[-1] +'.h5'
        mus_path = osp.join(base_path, mus_path)
        # .h5 경로 기반 Music Feature 추가 반영
        music_fea = torch.zeros((video_length, 4800), dtype=torch.float32)
        if osp.exists(mus_path):
            try:
                with h5py.File(mus_path, "r") as f:
                    m = f["music"][:]          # (1, 640, 4800) or (T, 4800)
                if m.ndim == 3 and m.shape[0] == 1:
                    m = m.squeeze(0)  # (640, 4800)
                if m.ndim == 2:
                    if m.shape[1] == 4800:
                        m = m.T  # (T, 4800)
                music_fea = torch.from_numpy(m).float()     # (T, 4800)
                print(f"✓ H5 loaded: music={music_fea.shape}")
            except Exception as e:
                print(f"Failed to load H5 file: {mus_path}, error: {e}")
                

        clip_length = min(video_length, (self.n_sample_frames - 1) * self.sample_frame_rate + 1)

        # print("-------------------------------")
        # print(clip_length)
        # print(video_length)
        # print(type(random))
        #print("-------------------------------")

        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(
            start_idx, start_idx + clip_length - 1, self.n_sample_frames, dtype=int
        ).tolist()
        all_indices = list(range(0, video_length))
        available_indices = [i for i in all_indices if i not in batch_index]

        # ---------- reference frame index ---------- #
        reference_frame_idx = None
        if available_indices:
            reference_frame_idx = random.choice(available_indices)
        else:
            print("There is no available frame")
            extreme_sample_frame_rate = 2
            extreme_clip_length = min(video_length, (self.n_sample_frames - 1) * extreme_sample_frame_rate + 1)
            extreme_start_idx = random.randint(0, video_length - extreme_clip_length)
            extreme_batch_index = np.linspace(
                extreme_start_idx, extreme_start_idx + extreme_clip_length - 1, self.n_sample_frames, dtype=int
            ).tolist()
            extreme_available_indices = [i for i in all_indices if i not in extreme_batch_index]
            if extreme_available_indices:
                reference_frame_idx = random.choice(extreme_available_indices)
            else:
                print("There is no available frame in the extreme circumstance")
                print(frames_path)
                print(1 / 0)


        pose_pil_image_list = []
        tgt_pil_image_list = []
        # tgt_face_masks_list = []
        mus_feat_list = []   # 오디오 프레임별 feature를 쌓을 리스트

        reference_frame_path = osp.join(frames_path, frames_list[reference_frame_idx])
        reference_pil_image = Image.open(reference_frame_path).convert('RGB')
        reference_pil_image = self.resize_and_center_crop(reference_pil_image, self.width, self.height)
        reference_pil_image = torch.from_numpy(np.array(reference_pil_image)).float()
        reference_pil_image = reference_pil_image / 127.5 - 1

        seg_mask_name = seg_masks_path  # 또는 첫 프레임 이름 사용
        seg_mask_path = osp.join(seg_masks_path, seg_mask_name)
        
        try:
            seg_mask = Image.open(seg_mask_path).convert('L')
            seg_mask = self.resize_and_center_crop(seg_mask, self.width, self.height)
            seg_mask = torch.from_numpy(np.array(seg_mask)).float()
            seg_mask = seg_mask / 255.0
        except Exception as e:
            print(f"Failed loading seg mask: {seg_mask_path}, error: {e}")
            seg_mask = torch.ones(self.height, self.width)


        pose_pil_image_list = []
        tgt_pil_image_list = []

        for index in batch_index:
            tgt_img_path = osp.join(frames_path, frames_list[index])
            img_basename = osp.splitext(osp.basename(tgt_img_path))[0]
            frame_number = int(img_basename.lstrip('0')) if img_basename != '0' else 0
            pose_name = f"frame_{frame_number}.png"
            pose_path = osp.join(poses_path, pose_name)

            # --- video ---
            try:
                tgt_img_pil = Image.open(tgt_img_path).convert('RGB')
                tgt_img_pil = self.resize_and_center_crop(tgt_img_pil, self.width, self.height)
                tgt_img_tensor = torch.from_numpy(np.array(tgt_img_pil)).float()
                tgt_img_normalized = tgt_img_tensor / 127.5 - 1
                tgt_pil_image_list.append(tgt_img_normalized)
            except Exception as e:
                print(f"Failed loading image: {tgt_img_path}, error: {e}")
                tgt_pil_image_list.append(torch.zeros(self.height, self.width, 3))

            # --- pose ---
            try:
                pose = Image.open(pose_path).convert('RGB')
                pose = self.resize_and_center_crop(pose, self.width, self.height)
                pose = torch.from_numpy(np.array(pose)).float()
                pose = pose / 127.5 - 1
            except Exception as e:
                print(f"Failed loading pose: {pose_path}, error: {e}")
                pose = torch.zeros(self.height, self.width, 3)
            pose_pil_image_list.append(pose)

        music_selected = music_fea[batch_index]

        pose_pil_image_list = torch.stack(pose_pil_image_list, dim=0)
        tgt_pil_image_list = torch.stack(tgt_pil_image_list, dim=0)

        tgt_pil_image_list = rearrange(tgt_pil_image_list, "f h w c -> f c h w")
        reference_pil_image = rearrange(reference_pil_image, "h w c -> c h w")
        pose_pil_image_list = rearrange(pose_pil_image_list, "f h w c -> f c h w")


        T = tgt_pil_image_list.shape[0]
        print(T)
        # crop_len에 따라 자르기 또는 패딩
        # if self.crop_len is not None and T >= self.crop_len:
        #     num_crops = T - self.crop_len + 1  
        #     pixel_crops = []
        #     pose_crops = []
        #     music_crops = []
        #     for start in range(num_crops):
        #         end = start + self.crop_len
        #         pixel_crops.append(tgt_pil_image_list[start:end])
        #         pose_crops.append(pose_pil_image_list[start:end])
        #         music_crops.append(music_selected[start:end])
            
        #     # (N, crop_len, C, H, W)
        #     tgt_pil_image_list = torch.stack(pixel_crops, dim=0)
        #     pose_pil_image_list = torch.stack(pose_crops, dim=0)
        #     music_selected = torch.stack(music_crops, dim=0)
        # elif self.crop_len is not None and T < self.crop_len:
        #     pad_len = self.crop_len - T
        #     tgt_pil_image_list = torch.cat([
        #         tgt_pil_image_list, 
        #         torch.zeros(pad_len, *tgt_pil_image_list.shape[1:])
        #     ], dim=0).unsqueeze(0)  # (1, crop_len, C, H, W)
        #     pose_pil_image_list = torch.cat([
        #         pose_pil_image_list,
        #         torch.zeros(pad_len, *pose_pil_image_list.shape[1:])
        #     ], dim=0).unsqueeze(0)
        #     music_selected = torch.cat([
        #         music_selected,
        #         torch.zeros(pad_len, *music_selected.shape[1:])
        #     ], dim=0).unsqueeze(0)
        # else:
        #     # crop_len이 None이면 그대로
        #     tgt_pil_image_list = tgt_pil_image_list.unsqueeze(0)
        #     pose_pil_image_list = pose_pil_image_list.unsqueeze(0)
        #     music_selected = music_selected.unsqueeze(0)

        reference_frame_id_ante_embedding = None
        sample = dict(
            pixel_values=tgt_pil_image_list,
            reference_image=reference_pil_image,
            pose_pixels=pose_pil_image_list,
            faceid_embeds=reference_frame_id_ante_embedding,
            seg_mask=seg_mask, 
            music_fea=music_selected,  
        )
        return sample

if __name__ == "__main__":
    print("=" * 70)
    print("Testing LargeScaleMusicVideos Dataset")
    print("=" * 70)

    # 데이터셋 초기화
    dataset = LargeScaleMusicVideos(
        root_path="/root/dataset/rec/gJS_sBM_cAll_d01_mJS0_ch09",
        txt_path="/root/dataset/video_list.txt",
        width=512,
        height=512,
        n_sample_frames=16,
        sample_frame_rate=2,
        crop_len=12,  # H5 파일 크롭 길이
        app=None,  # insightface app
        handler_ante=None,  # face embedding handler
        face_helper=None  # face helper
    )

    print(f"\n📊 Dataset Info:")
    print(f"   - Total videos: {len(dataset)}")
    print(f"   - Sample frames: {dataset.n_sample_frames}")
    print(f"   - Frame rate: {dataset.sample_frame_rate}")
    print(f"   - Output size: {dataset.width}x{dataset.height}")
    if dataset.crop_len:
        print(f"   - H5 crop length: {dataset.crop_len}")

    # 첫 번째 아이템 테스트
    print("\n" + "=" * 70)
    print("Loading first sample...")
    print("=" * 70)

    try:
        sample = dataset[0]

        print(f"\n✓ Successfully loaded sample from: {dataset.video_files[0]}")
        print(f"\n📦 Sample contains {len(sample)} keys:")

        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"\n   [{key}]")
                print(f"      - Shape: {value.shape}")
                print(f"      - Dtype: {value.dtype}")
                print(f"      - Range: [{value.min():.3f}, {value.max():.3f}]")
                print(f"      - Mean: {value.mean():.3f}")
            elif isinstance(value, np.ndarray):
                print(f"\n   [{key}]")
                print(f"      - Shape: {value.shape}")
                print(f"      - Dtype: {value.dtype}")
            else:
                print(f"\n   [{key}]: {type(value)}")

    except Exception as e:
        print(f"\n✗ Error loading sample: {e}")
        import traceback
        traceback.print_exc()
