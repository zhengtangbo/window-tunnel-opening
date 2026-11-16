import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
from contextlib import ExitStack
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from render_opening import FPS, OPEN_END, REFERENCE_FRAMES, WAVE_END, WAVE_START, Opening
from road_alignment import road_for_frame, road_from_points, select_track


WINDOWS = "透视窗格与踏板.mov"
LIGHT = "光束与揭开亮边.mov"
BASE_VIDEO = "底轨.mp4"
MANIFEST = "opening.json"
FFMPEG = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]


def probe(path):
    result = subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)])
    return json.loads(result)["streams"]


def size_for(stream, limit):
    sar = (stream.get("sample_aspect_ratio") or "1:1").split(":")
    ratio = int(sar[0]) / int(sar[1]) if len(sar) == 2 and all(int(part) for part in sar) else 1
    width, height = stream["width"] * ratio, stream["height"]
    rotation = next((data["rotation"] for data in stream.get("side_data_list", []) if "rotation" in data), 0)
    if round(rotation) % 180 == 90:
        width, height = height, width
    scale = min(1, limit / max(width, height))
    return tuple(max(4, 2 * round(dimension * scale / 2)) for dimension in (width, height))


def prepare_base(source, target, start, seconds, mute):
    streams = probe(source)
    video = next((s for s in streams if s["codec_type"] == "video" and not s.get("disposition", {}).get("attached_pic")), None)
    if video is None:
        raise ValueError("The input does not contain a video stream")
    width, height = size_for(video, 3840)
    filters = []
    if video.get("color_transfer") in {"smpte2084", "arib-std-b67"}:
        filters += [
            "zscale=t=linear:npl=100", "format=gbrpf32le", "zscale=p=bt709",
            "tonemap=tonemap=mobius:desat=2", "zscale=t=bt709:m=bt709:r=limited",
        ]
    filters += [f"scale={width}:{height}:flags=lanczos:out_color_matrix=bt709", "setsar=1", f"fps={FPS}"]
    command = [
        *FFMPEG, "-threads", "4", "-ss", str(start), "-i", str(source),
        "-map", f"0:{video['index']}", "-t", str(seconds), "-filter_threads", "4",
        "-vf", ",".join(filters), "-c:v", "libx264", "-preset", "fast", "-crf", "16",
        "-threads", "5", "-pix_fmt", "yuv420p", "-colorspace", "bt709",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-movflags", "+faststart",
    ]
    if mute:
        command += ["-an"]
    else:
        command += ["-map", "0:a:0?", "-af", "aresample=async=1", "-c:a", "aac", "-b:a", "256k"]
    subprocess.run([*command, str(target)], check=True)
    output = next((s for s in probe(target) if s["codec_type"] == "video"), {})
    frames = int(output.get("nb_frames", 0))
    if frames < 3:
        raise ValueError("This opening needs at least three usable frames after the requested start")
    return output, frames


def base_frame(base, frame_number, width, height):
    raw = subprocess.check_output([
        *FFMPEG, "-threads", "4", "-i", str(base), "-map", "0:v:0", "-frames:v", "1",
        "-vf", f"select=eq(n\\,{frame_number}),scale={width}:{height}", "-f", "rawvideo",
        "-pix_fmt", "bgr24", "-threads", "1", "-",
    ])
    return np.frombuffer(raw, np.uint8).reshape(height, width, 3)


def tread_texture(base, frame, width, height, point, span=None):
    image = base_frame(base, frame, width, height)
    x, y = point
    patch_width = max(2, round(width * min(0.09, span))) if span is not None else max(2, round(width * 0.09))
    patch_height = max(2, round(height * 0.054))
    left = min(width - patch_width, max(0, round(width * x - patch_width / 2)))
    top = min(height - patch_height, max(0, round(height * y - patch_height / 2)))
    patch = image[top:top + patch_height, left:left + patch_width]
    return cv2.resize(patch, (96, 42), interpolation=cv2.INTER_AREA)


def find_road(base, motion_end, size, points, disabled, folder):
    if disabled:
        return [], "off"
    if points:
        return [(0, road_from_points(points))], "manual"
    # Sample through the reveal, including the last few frames of the transition.
    indices = sorted({round((motion_end - 1) * ref / WAVE_END)
                      for ref in (145, 160, 174, WAVE_START, 188, 195, 201, WAVE_END)})
    if len(indices) < 2:
        return [], "unconfirmed"
    width, height = size
    scale = min(1, 640 / max(width, height))
    width, height = max(4, round(width * scale)), max(4, round(height * scale))
    selected = "+".join(f"eq(n\\,{frame})" for frame in indices)
    raw = subprocess.check_output([
        *FFMPEG, "-threads", "4", "-i", str(base), "-map", "0:v:0",
        "-vf", f"select={selected},scale={width}:{height}", "-frames:v", str(len(indices)),
        "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-threads", "1", "-",
    ])
    images = np.frombuffer(raw, np.uint8).reshape(len(indices), height, width, 3)
    track = select_track(list(zip(indices, images)))
    if not track:
        return [], "unconfirmed"
    frame, road = min(track, key=lambda match: abs(match[0] - (motion_end - 1) * 195 / WAVE_END))
    diagnostic = images[indices.index(frame)].copy()
    rows = road.edges[:, 0] * height
    for x in [road.edges[:, 1], road.edges[:, 2], road.edges[:, 1:].mean(axis=1)]:
        line = np.rint(np.column_stack((x * width, rows))).astype(np.int32)
        cv2.polylines(diagnostic, [line], False, (32, 226, 255), 2, cv2.LINE_AA)
    cv2.imencode(".jpg", diagnostic)[1].tofile(folder / "road-check.jpg")
    return track, "auto"


def writer(path, width, height, alpha=False):
    command = [
        *FFMPEG, "-f", "rawvideo", "-pixel_format", "bgra" if alpha else "bgr24",
        "-video_size", f"{width}x{height}", "-framerate", str(FPS), "-i", "-",
        "-an", "-filter_threads", "2", "-colorspace", "bt709", "-color_primaries", "bt709",
        "-color_trc", "bt709", "-threads", "4",
    ]
    if alpha:
        command += [
            "-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le",
            "-alpha_bits", "8", "-tag:v", "ap4h",
        ]
    else:
        command += ["-c:v", "libx264", "-preset", "fast", "-crf", "16", "-pix_fmt", "yuv420p"]
    return subprocess.Popen([*command, str(path)], stdin=subprocess.PIPE)


def alpha_frame(image, alpha):
    # Jianying composites ProRes 4444 as premultiplied alpha.
    return np.dstack([
        np.rint(image * alpha[..., None]).astype(np.uint8),
        np.rint(alpha * 255).astype(np.uint8),
    ]).tobytes()


def render(folder, base, frames, size, point, road_points=None, no_road=False):
    width, height = size
    opening = Opening(width, height)
    motion_end = max(2, min(frames - 1, math.ceil(OPEN_END * frames / REFERENCE_FRAMES)))
    route, status = find_road(base, motion_end, size, road_points, no_road, folder)
    texture_frame = min(frames - 1, round((motion_end - 1) * 192 / WAVE_END))
    road = road_for_frame(route, texture_frame)
    span = None
    if not point and road is not None:
        point = road.texture_point()
        left, right = road.at(point[1])
        span = float((right - left) * 0.44)
    point = point or ((0.315, 0.835) if width >= height else (0.5, 0.81))
    opening.stone = tread_texture(base, texture_frame, width, height, point, span)
    expected_bytes = width * height * 3
    with ExitStack() as cleanup:
        reader = cleanup.enter_context(subprocess.Popen([
            *FFMPEG, "-threads", "4", "-i", str(base), "-an", "-filter_threads", "3",
            "-vf", f"scale={width}:{height}:flags=lanczos", "-threads", "2",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
        ], stdout=subprocess.PIPE))
        scene_writer = cleanup.enter_context(writer(folder / WINDOWS, width, height, alpha=True))
        light_writer = cleanup.enter_context(writer(folder / LIGHT, width, height, alpha=True))
        preview = cleanup.enter_context(writer(folder / "合成.mp4", width, height))
        for index in range(frames):
            data = reader.stdout.read(expected_bytes)
            if len(data) != expected_bytes:
                raise RuntimeError(f"Source decode ended on frame {index} of {frames}")
            frame = np.frombuffer(data, np.uint8).reshape(height, width, 3)
            if index < motion_end:
                sample = index * WAVE_END / (motion_end - 1)
                road = road_for_frame(route, index) if sample >= 4.45 * FPS else None
                scene, opacity, light, light_alpha, frame = opening.render(frame, sample, road)
                scene_writer.stdin.write(alpha_frame(scene, opacity))
                light_writer.stdin.write(alpha_frame(light, light_alpha))
                if index == motion_end - 1:
                    scene_writer.stdin.close()
                    light_writer.stdin.close()
            preview.stdin.write(frame.tobytes())
            if (index + 1) % FPS == 0 or index == frames - 1:
                print(f"{index + 1}/{frames}", flush=True)
        preview.stdin.close()
        if any(process.wait() for process in (reader, scene_writer, light_writer, preview)):
            raise RuntimeError("A decoder or encoder did not finish successfully")
    metadata = {"status": status, "samples": [
        {"frame": index, "edges": road.edges.round(5).tolist()} for index, road in route
    ]}
    return motion_end, metadata


def make(args):
    source = args.video.resolve(strict=True)
    folder = args.work_dir.resolve()
    output = args.output.resolve()
    if not source.is_file() or output == source or output.suffix.lower() != ".mp4":
        raise ValueError("Supply a source video and a separate .mp4 output path")
    if output.exists():
        raise FileExistsError(output)
    if folder.exists() and any(folder.iterdir()):
        raise ValueError(f"Use a new or empty working directory: {folder}")
    if not math.isfinite(args.start) or not math.isfinite(args.seconds) or args.start < 0 or args.seconds <= 0:
        raise ValueError("Start must be non-negative and duration must be positive")
    if args.texture and any(not 0 <= value <= 1 for value in args.texture):
        raise ValueError("Texture coordinates must be between 0 and 1")
    cv2.setNumThreads(3)
    folder.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    base = folder / BASE_VIDEO
    stream, frames = prepare_base(source, base, args.start, args.seconds, args.mute)
    size = size_for(stream, 1920)
    motion_end, road = render(folder, base, frames, size, args.texture, args.road_point, args.no_road)
    cover_frame = min(frames - 1, motion_end + 6)
    cover = base_frame(base, cover_frame, *size)
    cv2.imencode(".jpg", cover)[1].tofile(folder / "draft_cover.jpg")

    with TemporaryDirectory(dir=output.parent, prefix=".opening-") as temporary:
        candidate = Path(temporary) / "video.mp4"
        subprocess.run([
            *FFMPEG, "-i", str(folder / "合成.mp4"), "-i", str(base),
            "-map", "0:v:0", "-map", "1:a:0?", "-c", "copy", "-t", f"{frames / FPS:.8f}",
            "-movflags", "+faststart", str(candidate),
        ], check=True)
        output_stream = probe(candidate)[0]
        if int(output_stream["nb_frames"]) != frames:
            raise RuntimeError("Unexpected frame count in the rendered video")
        if output.exists():
            raise FileExistsError(output)
        shutil.move(candidate, output)
    manifest = {
        "source_name": source.name, "fps": FPS, "frames": frames, "motion_frames": motion_end,
        "width": stream["width"], "height": stream["height"], "preview_width": size[0], "preview_height": size[1],
        "start_seconds": args.start, "preview": str(output), "road_alignment": road,
    }
    (folder / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**manifest, "road_alignment": road["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render the window-tunnel opening and separate Jianying layers.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--seconds", type=float, default=9)
    parser.add_argument("--mute", action="store_true")
    parser.add_argument("--texture", type=float, nargs=2, metavar=("X", "Y"))
    road_options = parser.add_mutually_exclusive_group()
    road_options.add_argument("--road-point", type=float, nargs=3, action="append", metavar=("X", "Y", "WIDTH"))
    road_options.add_argument("--no-road", action="store_true", help="Keep the original tread motion")
    make(parser.parse_args())
