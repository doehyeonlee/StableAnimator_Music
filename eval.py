#!/usr/bin/env python3
# evaluate_fluency.py

"""
Temporal Fluency Metric for Video Generation Evaluation

이 스크립트는 세그멘테이션 기반 시간적 유창성(Temporal Fluency)을 측정합니다.

수식:
    c_t = |B_t ⊕ B_{t+1}| / |B_t ∪ B_{t+1}|  (프레임 변화 비율)
    μ_V = mean(c_t)  (평균 변화량)
    σ²_V = var(c_t)  (변화량 분산)
    F_V = 1 / (1 + μ_V + σ²_V)  (유창성 점수)

점수 해석:
    - 0.90~1.00: 최상 (실제 비디오 수준)
    - 0.80~0.90: 우수 (자연스러움)
    - 0.70~0.80: 양호 (약간의 끊김)
    - 0.60~0.70: 보통 (눈에 띄는 끊김)
    - < 0.60: 불량 (심한 끊김)

Author: [Your Name]
Date: 2025-01-28
"""

import numpy as np
import os
import glob
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import pandas as pd
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class FluencyEvaluator:
    """비디오 유창성 평가 클래스"""
    
    def __init__(self, fps: int = 30, window_seconds: int = 5):
        """
        Args:
            fps: 프레임레이트 (frames per second)
            window_seconds: 평가 윈도우 크기 (초)
        """
        self.fps = fps
        self.window_seconds = window_seconds
        self.window_frames = fps * window_seconds
    
    @staticmethod
    def load_segmentation(npy_path: str) -> Optional[np.ndarray]:
        """
        세그멘테이션 .npy 파일 로드
        
        Args:
            npy_path: .npy 파일 경로
            
        Returns:
            body_mask: 신체 영역 이진 마스크 (H, W) 또는 None
        """
        try:
            seg = np.load(npy_path)
            # 배경(0) 제외, 신체 부위만 (>0)
            body_mask = seg > 0
            return body_mask
        except Exception as e:
            print(f"Error loading {npy_path}: {e}")
            return None
    
    @staticmethod
    def compute_frame_change(mask1: np.ndarray, mask2: np.ndarray) -> Optional[float]:
        """
        두 프레임 간 신체 픽셀 변화 비율 계산
        
        수식: c_t = |B_t ⊕ B_{t+1}| / |B_t ∪ B_{t+1}|
        
        Args:
            mask1: 프레임 t의 신체 마스크
            mask2: 프레임 t+1의 신체 마스크
            
        Returns:
            change_ratio: 변화 비율 [0, 1]
        """
        if mask1 is None or mask2 is None:
            return None
        
        # XOR 연산: 변화된 픽셀 (대칭 차집합)
        changed_pixels = np.logical_xor(mask1, mask2)
        
        # Union: 전체 신체 픽셀
        total_body_pixels = np.logical_or(mask1, mask2).sum()
        
        if total_body_pixels == 0:
            return 0.0
        
        change_ratio = changed_pixels.sum() / total_body_pixels
        return float(change_ratio)
    
    def compute_temporal_fluency(
        self, 
        seg_files: List[str]
    ) -> Tuple[List[float], List[Dict], float, float, float]:
        """
        시간적 유창성 계산
        
        Args:
            seg_files: 정렬된 세그멘테이션 파일 경로 리스트
            
        Returns:
            frame_changes: 프레임별 변화량 [c_1, c_2, ..., c_{N-1}]
            window_fluency: 윈도우별 유창성 정보
            overall_fluency: 전체 유창성 점수 F_V
            avg_change: 평균 변화량 μ_V
            variance: 변화량 분산 σ²_V
        """
        # 1. 프레임별 변화량 계산
        frame_changes = []
        prev_mask = None
        
        print("프레임별 변화량 계산 중...")
        for seg_file in tqdm(seg_files, desc="Processing frames"):
            curr_mask = self.load_segmentation(seg_file)
            
            if prev_mask is not None and curr_mask is not None:
                change = self.compute_frame_change(prev_mask, curr_mask)
                if change is not None:
                    frame_changes.append(change)
            
            prev_mask = curr_mask
        
        if len(frame_changes) == 0:
            print("⚠️ 유효한 프레임 변화를 계산할 수 없습니다.")
            return [], [], 0.0, 0.0, 0.0
        
        # 2. 전체 통계 계산
        avg_change = float(np.mean(frame_changes))
        variance = float(np.var(frame_changes))
        overall_fluency = 1.0 / (1.0 + avg_change + variance)
        
        # 3. 윈도우별 유창성 계산
        window_fluency = []
        
        print(f"\n{self.window_seconds}초 윈도우별 유창성 계산 중...")
        for i in range(0, len(frame_changes), self.window_frames):
            window = frame_changes[i:i+self.window_frames]
            
            if len(window) > 0:
                w_avg = float(np.mean(window))
                w_var = float(np.var(window))
                w_fluency = 1.0 / (1.0 + w_avg + w_var)
                
                window_fluency.append({
                    'window_id': len(window_fluency) + 1,
                    'start_frame': i,
                    'end_frame': min(i + self.window_frames, len(frame_changes)),
                    'num_frames': len(window),
                    'avg_change': w_avg,
                    'variance': w_var,
                    'fluency_score': w_fluency
                })
        
        return frame_changes, window_fluency, overall_fluency, avg_change, variance
    
    def evaluate_folder(
        self, 
        seg_folder: str, 
        output_dir: Optional[str] = None,
        verbose: bool = True
    ) -> Optional[Dict]:
        """
        폴더 내 모든 세그멘테이션 평가
        
        Args:
            seg_folder: 세그멘테이션 폴더 경로
            output_dir: 결과 저장 디렉토리 (None이면 저장 안 함)
            verbose: 상세 출력 여부
            
        Returns:
            results: 평가 결과 딕셔너리
        """
        # .npy 파일 찾기 및 정렬
        seg_files = sorted(glob.glob(os.path.join(seg_folder, '*_seg.npy')))
        
        if len(seg_files) == 0:
            print(f"❌ {seg_folder}에서 세그멘테이션 파일을 찾을 수 없습니다.")
            return None
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"📁 폴더: {seg_folder}")
            print(f"📊 총 프레임 수: {len(seg_files)}")
            print(f"🎬 FPS: {self.fps}")
            print(f"⏱️  윈도우: {self.window_seconds}초")
            print('='*70)
        
        # 유창성 계산
        frame_changes, window_fluency, overall_fluency, avg_change, variance = \
            self.compute_temporal_fluency(seg_files)
        
        if len(frame_changes) == 0:
            return None
        
        # 결과 출력
        if verbose:
            self._print_results(
                overall_fluency, avg_change, variance, 
                frame_changes, window_fluency
            )
        
        # 결과 저장
        folder_name = Path(seg_folder).parent.parent.name
        # folder_name = Path(seg_folder).parent.name
        
        results = {
            'folder': seg_folder,
            'folder_name': folder_name,
            'total_frames': len(seg_files),
            'fps': self.fps,
            'window_seconds': self.window_seconds,
            'overall_fluency': overall_fluency,
            'avg_change': avg_change,
            'variance': variance,
            'min_change': float(np.min(frame_changes)),
            'max_change': float(np.max(frame_changes)),
            'median_change': float(np.median(frame_changes)),
            'std_change': float(np.std(frame_changes)),
            'frame_changes': frame_changes,
            'window_fluency': window_fluency,
            'grade': self._get_grade(overall_fluency)
        }
        
        if output_dir:
            self._save_results(results, output_dir, folder_name)
        
        return results
    
    def _print_results(
        self, 
        overall_fluency: float,
        avg_change: float, 
        variance: float,
        frame_changes: List[float], 
        window_fluency: List[Dict]
    ):
        """결과 출력"""
        print("\n" + "="*70)
        print("📈 평가 결과")
        print("="*70)
        print(f"전체 유창성 점수 (F_V): {overall_fluency:.4f}")
        print(f"평균 변화량 (μ_V): {avg_change:.4f}")
        print(f"변화량 분산 (σ²_V): {variance:.4f}")
        print(f"변화량 범위: [{np.min(frame_changes):.4f}, {np.max(frame_changes):.4f}]")
        print(f"변화량 중앙값: {np.median(frame_changes):.4f}")
        print(f"변화량 표준편차: {np.std(frame_changes):.4f}")
        print(f"등급: {self._get_grade(overall_fluency)}")
        
        print(f"\n{self.window_seconds}초 윈도우별 유창성:")
        for w in window_fluency:
            print(f"  윈도우 {w['window_id']} "
                  f"(프레임 {w['start_frame']}-{w['end_frame']}): "
                  f"F={w['fluency_score']:.4f}, "
                  f"μ={w['avg_change']:.4f}, "
                  f"σ²={w['variance']:.4f}")
    
    @staticmethod
    def _get_grade(fluency: float) -> str:
        """유창성 점수를 등급으로 변환"""
        if fluency >= 0.90:
            return "Excellent"
        elif fluency >= 0.80:
            return "Good"
        elif fluency >= 0.70:
            return "Fair"
        elif fluency >= 0.60:
            return "Poor"
        else:
            return "Very Poor"
    
    def _save_results(self, results: Dict, output_dir: str, folder_name: str):
        """결과 저장"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. JSON 저장 (상세 정보)
        json_file = os.path.join(output_dir, f"{folder_name}_fluency.json")
        
        # frame_changes는 용량이 크므로 별도 저장 옵션
        json_results = results.copy()
        if len(results['frame_changes']) > 1000:
            # 프레임이 많으면 통계만 저장
            json_results['frame_changes'] = {
                'note': 'Too many frames. Statistics only.',
                'count': len(results['frame_changes']),
                'sample_first_10': results['frame_changes'][:10],
                'sample_last_10': results['frame_changes'][-10:]
            }
        
        with open(json_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n✅ JSON 저장: {json_file}")
        
        # 2. CSV 저장/업데이트 (요약 정보)
        csv_file = os.path.join(output_dir, "fluency_summary.csv")
        self._update_csv(csv_file, results)
        print(f"✅ CSV 업데이트: {csv_file}")
    
    @staticmethod
    def _update_csv(csv_file: str, results: Dict):
        """CSV 파일 업데이트 (기존 행 덮어쓰기 또는 새 행 추가)"""
        folder_name = results['folder_name']
        
        new_row = {
            'Folder': folder_name,
            'Overall_Fluency': f"{results['overall_fluency']:.4f}",
            'Avg_Change': f"{results['avg_change']:.4f}",
            'Variance': f"{results['variance']:.4f}",
            'Min_Change': f"{results['min_change']:.4f}",
            'Max_Change': f"{results['max_change']:.4f}",
            'Median_Change': f"{results['median_change']:.4f}",
            'Std_Change': f"{results['std_change']:.4f}",
            'Frames': results['total_frames'],
            'FPS': results['fps'],
            'Window_Sec': results['window_seconds'],
            'Grade': results['grade']
        }
        
        # CSV 파일 존재 여부 확인
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            
            # 같은 폴더가 이미 있는지 확인
            if folder_name in df['Folder'].values:
                # 기존 행 업데이트
                for col, value in new_row.items():
                    if col in df.columns:
                        df.loc[df['Folder'] == folder_name, col] = value
                    else:
                        df[col] = None
                        df.loc[df['Folder'] == folder_name, col] = value
                print(f"   ℹ️  기존 행 업데이트: {folder_name}")
            else:
                # 새로운 행 추가
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                print(f"   ➕ 새로운 행 추가: {folder_name}")
        else:
            # 새 CSV 파일 생성
            df = pd.DataFrame([new_row])
            print(f"   📄 새 CSV 파일 생성")
        
        # CSV 저장
        df.to_csv(csv_file, index=False)
    
    def batch_evaluate(
        self, 
        parent_folder: str, 
        output_dir: Optional[str] = None,
        pattern: str = '*/images/sapiens_seg'
    ):
        """
        여러 폴더 일괄 평가
        
        Args:
            parent_folder: 부모 폴더 경로
            output_dir: 결과 저장 디렉토리
            pattern: 세그멘테이션 폴더 검색 패턴
        """
        # sapiens_seg 폴더 찾기
        seg_folders = glob.glob(os.path.join(parent_folder, pattern))
        
        if len(seg_folders) == 0:
            print(f"❌ {parent_folder}에서 '{pattern}' 폴더를 찾을 수 없습니다.")
            return
        
        print(f"\n{'█'*70}")
        print(f"🔍 발견된 폴더: {len(seg_folders)}개")
        print('█'*70)
        
        all_results = []
        failed = []
        
        for i, seg_folder in enumerate(seg_folders, 1):
            print(f"\n\n{'='*70}")
            print(f"[{i}/{len(seg_folders)}]")
            print('='*70)
            
            try:
                result = self.evaluate_folder(seg_folder, output_dir, verbose=True)
                if result:
                    all_results.append(result)
                else:
                    failed.append(seg_folder)
            except Exception as e:
                print(f"❌ 에러 발생: {e}")
                failed.append(seg_folder)
        
        # 전체 요약
        self._print_summary(all_results, failed)
    
    @staticmethod
    def _print_summary(all_results: List[Dict], failed: List[str]):
        """전체 요약 출력"""
        if not all_results:
            print("\n⚠️ 평가된 결과가 없습니다.")
            return
        
        print("\n\n" + "="*70)
        print("📊 전체 요약")
        print("="*70)
        
        fluency_scores = [r['overall_fluency'] for r in all_results]
        avg_changes = [r['avg_change'] for r in all_results]
        variances = [r['variance'] for r in all_results]
        
        print(f"총 평가: {len(all_results)}개")
        print(f"실패: {len(failed)}개")
        print()
        
        print("유창성 점수 (F_V):")
        print(f"  평균: {np.mean(fluency_scores):.4f}")
        print(f"  중앙값: {np.median(fluency_scores):.4f}")
        print(f"  표준편차: {np.std(fluency_scores):.4f}")
        print(f"  범위: [{np.min(fluency_scores):.4f}, {np.max(fluency_scores):.4f}]")
        print()
        
        print("평균 변화량 (μ_V):")
        print(f"  평균: {np.mean(avg_changes):.4f}")
        print(f"  중앙값: {np.median(avg_changes):.4f}")
        print()
        
        print("분산 (σ²_V):")
        print(f"  평균: {np.mean(variances):.4f}")
        print(f"  중앙값: {np.median(variances):.4f}")
        print()
        
        # 등급별 분포
        grades = [r['grade'] for r in all_results]
        grade_counts = pd.Series(grades).value_counts()
        print("등급 분포:")
        for grade in ['Excellent', 'Good', 'Fair', 'Poor', 'Very Poor']:
            count = grade_counts.get(grade, 0)
            percentage = (count / len(all_results)) * 100
            print(f"  {grade}: {count}개 ({percentage:.1f}%)")
        
        if failed:
            print(f"\n⚠️ 실패한 폴더 {len(failed)}개:")
            for folder in failed[:10]:
                print(f"  - {folder}")
            if len(failed) > 10:
                print(f"  ... 외 {len(failed)-10}개")
        
        print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Sapiens 세그멘테이션 기반 Temporal Fluency 평가',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 단일 폴더 평가
  python evaluate_fluency.py /path/to/sapiens_seg --output ./results
  
  # 여러 폴더 일괄 평가
  python evaluate_fluency.py /path/to/parent --batch --output ./results
  
  # FPS와 윈도우 크기 지정
  python evaluate_fluency.py /path/to/sapiens_seg --fps 60 --window 3
        """
    )
    
    parser.add_argument(
        'input', 
        help='세그멘테이션 폴더 또는 부모 폴더 경로'
    )
    parser.add_argument(
        '--fps', 
        type=int, 
        default=30, 
        help='프레임레이트 (기본: 30)'
    )
    parser.add_argument(
        '--window', 
        type=int, 
        default=5, 
        help='평가 윈도우 크기 (초, 기본: 5)'
    )
    parser.add_argument(
        '--output', 
        help='결과 저장 디렉토리'
    )
    parser.add_argument(
        '--batch', 
        action='store_true', 
        help='여러 폴더 일괄 처리'
    )
    parser.add_argument(
        '--pattern',
        default='*/images/sapiens_seg',
        help='배치 모드에서 폴더 검색 패턴 (기본: */images/sapiens_seg)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='최소 출력 모드'
    )
    
    args = parser.parse_args()
    
    # Evaluator 생성
    evaluator = FluencyEvaluator(fps=args.fps, window_seconds=args.window)
    
    # 평가 실행
    if args.batch:
        evaluator.batch_evaluate(
            args.input, 
            args.output,
            pattern=args.pattern
        )
    else:
        evaluator.evaluate_folder(
            args.input, 
            args.output,
            verbose=not args.quiet
        )


if __name__ == '__main__':
    main()
