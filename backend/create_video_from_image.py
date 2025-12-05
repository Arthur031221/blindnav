#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
將照片轉換成30秒影片的腳本
每一幀都使用同一張照片
"""

import os
import cv2
import numpy as np

# ========== 設定區 ==========
# 輸入照片路徑
INPUT_IMAGE_PATH = r"D:\blindnav_local\backend\image\bus.jpg"

# 輸出影片資料夾
OUTPUT_VIDEO_DIR = r"D:\blindnav_local\backend\videos"

# 輸出影片檔名
OUTPUT_VIDEO_NAME = "bus_30s.mp4"

# 影片設定
VIDEO_DURATION_SECONDS = 30  # 30秒
FPS = 30  # 每秒30幀
# ============================

def create_video_from_image():
    """將照片轉換成30秒影片"""
    
    # 檢查輸入照片是否存在
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"❌ 錯誤：找不到輸入照片：{INPUT_IMAGE_PATH}")
        return False
    
    # 創建輸出資料夾（如果不存在）
    os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
    print(f"✓ 輸出資料夾已準備：{OUTPUT_VIDEO_DIR}")
    
    # 讀取照片
    print(f"📷 正在讀取照片：{INPUT_IMAGE_PATH}")
    image = cv2.imread(INPUT_IMAGE_PATH)
    
    if image is None:
        print(f"❌ 錯誤：無法讀取照片檔案：{INPUT_IMAGE_PATH}")
        return False
    
    # 取得照片尺寸
    height, width = image.shape[:2]
    print(f"✓ 照片尺寸：{width} x {height}")
    
    # 確保尺寸是偶數（某些編碼器要求）
    if width % 2 != 0:
        width -= 1
        image = image[:, :width]
        print(f"  調整寬度為偶數：{width}")
    if height % 2 != 0:
        height -= 1
        image = image[:height, :]
        print(f"  調整高度為偶數：{height}")
    
    print(f"✓ 最終尺寸：{width} x {height}")
    
    # 計算總幀數
    total_frames = VIDEO_DURATION_SECONDS * FPS
    print(f"📹 將創建 {VIDEO_DURATION_SECONDS} 秒的影片（{total_frames} 幀，{FPS} FPS）")
    
    # 設定輸出影片路徑
    output_path = os.path.join(OUTPUT_VIDEO_DIR, OUTPUT_VIDEO_NAME)
    
    # 如果檔案已存在，先刪除
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"✓ 已刪除舊的影片檔案")
    
    # 嘗試多種編碼器（按優先順序）
    codecs_to_try = [
        ('mp4v', '.mp4'),  # MPEG-4 編碼
        ('XVID', '.avi'),  # XVID 編碼（更通用）
        ('MJPG', '.avi'),  # Motion JPEG
    ]
    
    out = None
    used_codec = None
    used_ext = None
    
    for codec_name, ext in codecs_to_try:
        # 如果副檔名不匹配，調整輸出路徑
        test_path = output_path
        if not output_path.lower().endswith(ext.lower()):
            base_name = os.path.splitext(OUTPUT_VIDEO_NAME)[0]
            test_path = os.path.join(OUTPUT_VIDEO_DIR, f"{base_name}{ext}")
            if os.path.exists(test_path):
                os.remove(test_path)
        
        print(f"🔧 嘗試使用編碼器：{codec_name} ({ext})")
        fourcc = cv2.VideoWriter_fourcc(*codec_name)
        out = cv2.VideoWriter(test_path, fourcc, FPS, (width, height))
        
        if out.isOpened():
            used_codec = codec_name
            used_ext = ext
            output_path = test_path
            print(f"✓ 成功使用編碼器：{codec_name}")
            break
        else:
            out.release()
            out = None
    
    if out is None or not out.isOpened():
        print(f"❌ 錯誤：無法創建影片檔案，所有編碼器都失敗")
        print(f"   請檢查 OpenCV 是否正確安裝")
        return False
    
    # 將同一張照片寫入每一幀
    print("🎬 正在生成影片...")
    for frame_num in range(total_frames):
        out.write(image)
        
        # 顯示進度（每10%顯示一次）
        if (frame_num + 1) % (total_frames // 10) == 0:
            progress = ((frame_num + 1) / total_frames) * 100
            print(f"   進度：{progress:.0f}% ({frame_num + 1}/{total_frames} 幀)")
    
    # 釋放資源
    out.release()
    
    # 檢查檔案是否成功創建
    if not os.path.exists(output_path):
        print(f"❌ 錯誤：影片檔案創建失敗")
        return False
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    
    # 驗證影片是否可以正常讀取
    print("\n🔍 正在驗證影片檔案...")
    test_cap = cv2.VideoCapture(output_path)
    if not test_cap.isOpened():
        print(f"❌ 警告：生成的影片檔案無法被 OpenCV 讀取")
        test_cap.release()
        return False
    
    # 讀取第一幀驗證
    ret, test_frame = test_cap.read()
    test_cap.release()
    
    if not ret or test_frame is None:
        print(f"❌ 警告：無法讀取影片內容")
        return False
    
    # 檢查影片資訊
    test_cap = cv2.VideoCapture(output_path)
    actual_fps = test_cap.get(cv2.CAP_PROP_FPS)
    actual_frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    test_cap.release()
    
    print(f"✅ 影片驗證成功！")
    print(f"\n📊 影片資訊：")
    print(f"   檔案路徑：{output_path}")
    print(f"   檔案大小：{file_size:.2f} MB")
    print(f"   編碼格式：{used_codec} ({used_ext})")
    print(f"   影片長度：{VIDEO_DURATION_SECONDS} 秒")
    print(f"   解析度：{width} x {height}")
    print(f"   設定幀率：{FPS} FPS")
    print(f"   實際幀率：{actual_fps:.2f} FPS")
    print(f"   總幀數：{actual_frame_count} 幀")
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("照片轉影片工具")
    print("=" * 60)
    print()
    
    success = create_video_from_image()
    
    print()
    if success:
        print("🎉 完成！")
    else:
        print("❌ 失敗！請檢查錯誤訊息。")
    
    print("=" * 60)

