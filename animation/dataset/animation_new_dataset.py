import os
import os.path as osp
import random
import warnings

import numpy as np
import torch
import cv2
import h5py
from PIL import Image
from torch.utils.data.dataset import Dataset
from einops import rearrange
from animation.modules.face_model import FaceModel



class LargeScaleMusicVideos(Dataset):
    def __init__(self, root_path, txt_path, width, height, n_sample_frames, sample_frame_rate, 
                 sample_margin=30, app=None, handler_ante=None, face_helper=None):
        self.root_path = root_path
        self.txt_path = txt_path
        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_frame_rate = sample_frame_rate
        self.sample_margin = sample_margin

        self.video_files = self._read_txt_file_images()

        self.app = app
        self.handler_ante = handler_ante
        self.face_helper = face_helper

    def _read_txt_file_images(self):
        with open(self.txt_path, 'r') as file:
            video_files = [line.strip() for line in file if line.strip()]
        return video_files

    def __len__(self):
        return len(self.video_files)

    def frame_count(self, frames_path):
        files = os.listdir(frames_path)
        image_files = [f for f in files if f.endswith(('.png', '.jpg'))]
        return len(image_files)

    def find_frames_list(self, frames_path):
        files = os.listdir(frames_path)
        image_files = [f for f in files if f.endswith(('.png', '.jpg'))]
        
        if image_files and image_files[0].startswith('frame_'):
            image_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        else:
            image_files.sort(key=lambda x: int(x.split('.')[0]))
        return image_files


    def get_face_masks(self, pil_img):
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
        """이미지를 비율 유지하며 리사이즈 후 center crop"""
        width, height = img.size
        
        # 타겟 사이즈보다 작으면 그냥 리사이즈
        if width <= target_width and height <= target_height:
            return img.resize((target_width, target_height), Image.LANCZOS)
        
        # 비율 계산 (짧은 쪽을 타겟에 맞춤)
        scale = max(target_width / width, target_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # 리사이즈
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Center crop
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        
        return img.crop((left, top, right, bottom))

    def __getitem__(self, idx):
        try:
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            warnings.filterwarnings('ignore', category=FutureWarning)
            
            base_path = '/root/aist_hdf5/rec/'
            video_base_path = self.video_files[idx]
            frames_path = osp.join(video_base_path, "images")
            #poses_path = osp.join(video_base_path, "poses")
            face_masks_path = osp.join(video_base_path, "faces")
            
            video_length = self.frame_count(frames_path)
            frames_list = self.find_frames_list(frames_path)
            
            # Music feature 로드
            mus_path = video_base_path.split('/')[-1] + '.h5'
            mus_path = osp.join(base_path, mus_path)
            
            music_fea = torch.zeros((video_length, 4800), dtype=torch.float32)
            if osp.exists(mus_path):
                try:
                    with h5py.File(mus_path, "r") as f:
                        m = f["music"][:]
                    
                    # Shape 정리
                    if m.ndim == 3 and m.shape[0] == 1:
                        m = m.squeeze(0)  # (1, T, 4800) -> (T, 4800)
                    
                    if m.ndim == 2:
                        # (T, 4800) 형태가 되도록 확인
                        if m.shape[1] != 4800 and m.shape[0] == 4800:
                            m = m.T  # (4800, T) -> (T, 4800)
                        # (720, 4800) 형태라면 그대로 유지 (시간축이 720)
                    
                    music_fea = torch.from_numpy(m).float()
                    # print(f"H5 loaded: music={music_fea.shape}")
                except Exception as e:
                    print(f"Failed to load H5 file: {mus_path}, error: {e}")

            # 클립 길이 계산
            clip_length = min(video_length, (self.n_sample_frames - 1) * self.sample_frame_rate + 1)
            start_idx = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(
                start_idx, start_idx + clip_length - 1, self.n_sample_frames, dtype=int
            ).tolist()

            # 참조 프레임 선택
            all_indices = list(range(video_length))
            available_indices = [i for i in all_indices if i not in batch_index]
            
            if available_indices:
                reference_frame_idx = random.choice(available_indices)
            else:
                extreme_sample_frame_rate = 2
                extreme_clip_length = min(video_length, (self.n_sample_frames - 1) * extreme_sample_frame_rate + 1)
                extreme_start_idx = random.randint(0, video_length - extreme_clip_length)
                extreme_batch_index = np.linspace(
                    extreme_start_idx, extreme_start_idx + extreme_clip_length - 1, 
                    self.n_sample_frames, dtype=int
                ).tolist()
                extreme_available_indices = [i for i in all_indices if i not in extreme_batch_index]
                
                if extreme_available_indices:
                    reference_frame_idx = random.choice(extreme_available_indices)
                else:
                    raise ValueError(f"No available reference frame in {frames_path}")

            # 참조 프레임 로드
            reference_frame_path = osp.join(frames_path, frames_list[reference_frame_idx])
            reference_pil_image = Image.open(reference_frame_path).convert('RGB')
            reference_pil_image = self.resize_and_center_crop(reference_pil_image, self.width, self.height)
            reference_pil_image = torch.from_numpy(np.array(reference_pil_image)).float()
            reference_pil_image = reference_pil_image / 127.5 - 1

            reference_frame_face_pil = Image.open(reference_frame_path).convert('RGB')
            reference_frame_face_pil = self.resize_and_center_crop(reference_frame_face_pil, self.width, self.height)
            reference_frame_face = np.array(reference_frame_face_pil)
            reference_frame_face_bgr = cv2.cvtColor(reference_frame_face, cv2.COLOR_RGB2BGR)
            reference_frame_face_info = self.app.get(reference_frame_face_bgr)
            if len(reference_frame_face_info) > 0:
                reference_frame_face_info = sorted(
                    reference_frame_face_info, 
                    key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1])
                )[-1]
                reference_frame_id_ante_embedding = reference_frame_face_info['embedding']
            else:
                reference_frame_id_ante_embedding = None

            # 타겟 프레임들 로드
    #        pose_pil_image_list = []
            tgt_pil_image_list = []
            tgt_face_masks_list = []

            for index in batch_index:
                tgt_img_path = osp.join(frames_path, frames_list[index])
                img_basename = osp.splitext(osp.basename(tgt_img_path))[0]
                frame_number = int(img_basename.lstrip('0')) if img_basename != '0' else 0
                
            #    pose_name = f"frame_{frame_number}.png"
            #    pose_path = osp.join(poses_path, pose_name)
                
                face_name = img_basename + '.png'
                face_mask_path = osp.join(face_masks_path, face_name)

                # 타겟 이미지
                try:
                    tgt_img_pil = Image.open(tgt_img_path).convert('RGB')
                    tgt_img_pil = self.resize_and_center_crop(tgt_img_pil, self.width, self.height)
                    tgt_img_tensor = torch.from_numpy(np.array(tgt_img_pil)).float()
                    tgt_img_normalized = tgt_img_tensor / 127.5 - 1
                    tgt_pil_image_list.append(tgt_img_normalized)
                except Exception as e:
                    print(f"Failed loading image: {tgt_img_path}, error: {e}")
                    tgt_pil_image_list.append(torch.zeros(self.height, self.width, 3))

                # Pose
                # try:
                #     pose = Image.open(pose_path).convert('RGB')
                #     pose = self.resize_and_center_crop(pose, self.width, self.height)
                #     pose = torch.from_numpy(np.array(pose)).float()
                #     pose = pose / 127.5 - 1
                # except Exception as e:
                #     print(f"Failed loading pose: {pose_path}, error: {e}")
                #     pose = torch.zeros(self.height, self.width, 3)
                # pose_pil_image_list.append(pose)

                # Face mask
                try:
                    face = Image.open(face_mask_path).convert('L')
                    face = self.resize_and_center_crop(face, self.width, self.height)
                    face = torch.from_numpy(np.array(face)).float()
                    face = face / 255.0  
                    face = face.unsqueeze(0)  # ✓ (H, W) -> (1, H, W)
                except Exception as e:
                    print(f"Failed loading face: {face_mask_path}, error: {e}")
                    face = torch.zeros(1, self.height, self.width)  
                tgt_face_masks_list.append(face)

            # Music feature 선택
            music_selected = music_fea[batch_index]  # (n_sample_frames, 4800)

            # 텐서 변환 및 차원 재배열
            tgt_pil_image_list = torch.stack(tgt_pil_image_list, dim=0)
            #pose_pil_image_list = torch.stack(pose_pil_image_list, dim=0)
            tgt_face_masks_list = torch.stack(tgt_face_masks_list, dim=0)

            # (F, H, W, C) -> (F, C, H, W)
            tgt_pil_image_list = rearrange(tgt_pil_image_list, "f h w c -> f c h w")
            reference_pil_image = rearrange(reference_pil_image, "h w c -> c h w")
            #pose_pil_image_list = rearrange(pose_pil_image_list, "f h w c -> f c h w")
            #tgt_face_masks_list = rearrange(tgt_face_masks_list, "f h w c -> f c h w")

            sample = dict(
                pixel_values=tgt_pil_image_list,         # (F, 3, H, W)
                reference_image=reference_pil_image,      # (3, H, W)
                #pose_pixels=pose_pil_image_list,          # (F, 3, H, W)
                faceid_embeds=reference_frame_id_ante_embedding,  # None
                tgt_face_masks=tgt_face_masks_list,       # (F, 3, H, W)                     # (H, W)
                music_fea=music_selected,                 # (F, 4800)
            )
            if (
                sample["pixel_values"] is None
                or sample["reference_image"] is None
                or sample["faceid_embeds"] is None
                or sample["tgt_face_masks"] is None
                or sample["music_fea"] is None
            ):
                raise ValueError(f"Invalid sample at idx={idx}, path={self.video_files[idx]}")

            return sample
        except Exception as e:
            print(f"[WARN] Skipped sample idx={idx}, path={self.video_files[idx]}, error={e}")
            return None


if __name__ == "__main__":
    print("=" * 70)
    print("Testing LargeScaleMusicVideos Dataset")
    print("=" * 70)
    face_model = FaceModel()
    dataset = LargeScaleMusicVideos(
        root_path="/root/aist_hdf5/rec",
        txt_path="/root/aist_hdf5/full_list.txt",
        width=512,
        height=512,
        n_sample_frames=16,
        sample_frame_rate=2,
        app=face_model.app,
        handler_ante=face_model.handler_ante,
        face_helper=face_model.face_helper
    )

    print(f"\nDataset Info:")
    print(f"   - Total videos: {len(dataset)}")
    print(f"   - Sample frames: {dataset.n_sample_frames}")
    print(f"   - Frame rate: {dataset.sample_frame_rate}")
    print(f"   - Output size: {dataset.width}x{dataset.height}")

    print("\n" + "=" * 70)
    print("Loading first sample...")
    print("=" * 70)

    errors = []
    for idx in range(len(dataset)):
        print(f"\n[{idx+1}/{len(dataset)}] Loading: {dataset.video_files[idx]}")
        try:
            sample = dataset[idx]
            # 각 key별 shape/type/range 출력
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"   [{key}] - shape: {tuple(value.shape)}, dtype: {value.dtype}, "
                          f"range: [{value.min():.3f}, {value.max():.3f}], mean: {value.mean():.3f}")
                elif isinstance(value, np.ndarray):
                    print(f"   [{key}] - shape: {value.shape}, dtype: {value.dtype}")
                else:
                    print(f"   [{key}] - type: {type(value)}")

            # validation 체크
            assert sample['pixel_values'].shape[0] == dataset.n_sample_frames
            assert sample['music_fea'].shape[0] == dataset.n_sample_frames
            assert sample['music_fea'].shape[1] == 4800

            print("   -> Validation passed")

        except Exception as e:
            print(f"   !! Error in {dataset.video_files[idx]}: {e}")
            import traceback
            traceback.print_exc()
            errors.append((dataset.video_files[idx], str(e)))

    print("\n" + "=" * 70)
    print("FINISHED VALIDATION")
    print("=" * 70)

    if errors:
        print(f"\n{len(errors)} errors found:")
        for f, msg in errors:
            print(f"   - {f}: {msg}")
    else:
        print("\nAll samples loaded and validated successfully!")