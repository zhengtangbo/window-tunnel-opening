import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
from tempfile import TemporaryDirectory
from uuid import uuid4

from make_opening import BASE_VIDEO, LIGHT, MANIFEST, WINDOWS


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def create(assets, root, name):
    candidates = [
        os.environ.get("JY_SKILL_ROOT"),
        Path(__file__).resolve().parents[2] / "jianying-editor",
        Path.home() / ".codex/skills/jianying-editor",
    ]
    skill = next((Path(p) for p in candidates if p and (Path(p) / "scripts/jy_wrapper.py").is_file()), None)
    if skill is None:
        raise FileNotFoundError("Locate the installed jianying-editor skill and set JY_SKILL_ROOT")
    sys.path.insert(0, str(skill / "scripts"))
    from jy_wrapper import JyProject

    if not name or name.endswith((".", " ")) or Path(name).name != name or any(c in name for c in '<>:"/\\|?*'):
        raise ValueError("Use a valid Windows folder name for the draft")
    target = root / name
    if target.exists():
        raise FileExistsError(target)
    plan = json.loads((assets / MANIFEST).read_text(encoding="utf-8"))
    duration = round(plan["frames"] * 1_000_000 / plan["fps"])
    motion = round(plan["motion_frames"] * 1_000_000 / plan["fps"])
    if not 0 < motion < duration:
        raise ValueError("Unexpected timing in the rendering manifest")
    root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=root, prefix=".window-tunnel-") as temporary:
        project = JyProject(name, width=plan["width"], height=plan["height"], drafts_root=temporary, overwrite=False)
        project._explicit_res = True
        project.script.fps = plan["fps"]
        folder = Path(project.draft_dir)
        materials = folder / "materials"
        materials.mkdir()
        for filename, display, track, length, volume in [
            (BASE_VIDEO, plan["source_name"], "原片与原声", duration, 1.0),
            (WINDOWS, WINDOWS, "透视窗格与踏板", motion, 0.0),
            (LIGHT, LIGHT, "光束与揭开亮边", motion, 0.0),
        ]:
            staged = materials / filename
            shutil.copyfile(assets / filename, staged)
            # Both the CFR base and the premultiplied MOVs are ready for direct import.
            segment = project._add_video_safe(str(staged), start_time=0, duration=length, track_name=track)
            if segment is None:
                raise RuntimeError(f"Could not import {filename}")
            segment.material_instance.material_name = display
            segment.volume = volume
        project.save()
        shutil.copyfile(assets / "draft_cover.jpg", folder / "draft_cover.jpg")
        identity = str(uuid4()).upper()
        content = folder / "draft_info.json"
        timeline = json.loads(content.read_text(encoding="utf-8-sig"))
        timeline.update(id=identity, name=name)
        tracks = timeline["tracks"]
        if [track["type"] for track in tracks] != ["video", "video", "video"]:
            raise RuntimeError("Expected three video tracks")
        expected = [duration, motion, motion]
        for track, length in zip(tracks, expected, strict=True):
            if len(track["segments"]) != 1 or track["segments"][0]["target_timerange"] != {"start": 0, "duration": length}:
                raise RuntimeError("A track has an unexpected duration")
        for material in timeline["materials"]["videos"]:
            original = Path(material["path"])
            relative = original.relative_to(folder)
            if not original.is_file():
                raise FileNotFoundError(original)
            material["path"] = str(target / relative)
        write(content, timeline)
        shutil.copyfile(content, folder / "draft_content.json")
        meta = folder / "draft_meta_info.json"
        metadata = json.loads(meta.read_text(encoding="utf-8-sig"))
        stamp = time.time_ns() // 1000
        metadata.update(
            draft_id=identity, draft_name=name, draft_cover="draft_cover.jpg",
            draft_fold_path=str(target), draft_root_path=str(root), tm_duration=duration,
            tm_draft_create=stamp, tm_draft_modified=stamp,
            draft_timeline_materials_size=sum(file.stat().st_size for file in materials.iterdir()),
        )
        write(meta, metadata)
        if target.exists():
            raise FileExistsError(target)
        folder.rename(target)
    print(f"Jianying draft: {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a new self-contained three-track Jianying draft.")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--draft-root", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    create(args.work_dir.resolve(strict=True), args.draft_root.resolve(), args.name)
