import os

from moviepy.editor import concatenate_videoclips
from moviepy.editor import VideoFileClip

# 비디오 파일 경로 리스트
VIDEO_DIR = "data/test_fsa/"
# Get all video files (mp4, MOV) in the directory.
video_paths = [
    os.path.join(VIDEO_DIR, f)
    for f in os.listdir(VIDEO_DIR)
    if f.endswith((".mp4", ".MOV"))
]
# Sort by name.
video_paths.sort()
print("video_paths:", video_paths)
# 각 비디오 파일을 VideoFileClip 객체로 로드하고 원본 해상도 출력
clips = []
for vp in video_paths:
    clip = VideoFileClip(vp)
    print(f"Original resolution of {os.path.basename(vp)}: "
          f"{clip.size[0]}x{clip.size[1]}")
    clips.append(clip)

# 비디오 클립들을 하나로 합치기
final_clip = concatenate_videoclips(clips)
SAVE_NAME = "test.mp4"
USE_RESIZE = False
# 해상도 조절
# 예제에서는 출력 비디오를 1280x720 해상도로 조절합니다.
if USE_RESIZE:
    new_sizes = [(640, 360)]  #, (320, 180)]
else:
    new_sizes = [final_clip.size]
for new_size in new_sizes:
    final_clip_resized = final_clip.resize(newsize=new_size)

    # 합쳐진 비디오를 .mp4 형식으로 저장, 해상도 조절 적용
    newsize_str = f"{new_size[0]}x{new_size[1]}"
    NEW_SAVE_NAME = SAVE_NAME.replace(".mp4", f"_{newsize_str}.mp4")
    save_path = os.path.join(VIDEO_DIR, NEW_SAVE_NAME)
    final_clip_resized.write_videofile(save_path)
