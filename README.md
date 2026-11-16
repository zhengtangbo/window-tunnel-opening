# 窗格隧道打开 · Window Tunnel Opening

一个用于 Codex 和剪映专业版的开场剪辑技能。用一条视频生成逐格点亮的透视窗格、向前推进的踏板和扫开画面的发光边缘，最后接回同一条素材的全屏画面。

- 默认从第 0 秒取前 9 秒。短片会压缩动效时间，保留全屏收尾。
- 保持源素材的横竖比例，以 30 fps 合成。默认保留原声，适合后续在剪映加配乐。
- 从片头到揭开跟踪实景道路的边线和弯向。踏板从出现起就由同一帧路面生成，两侧窗格和光束随道路透视布置。
- 输出 MP4 和两个独立的透明动效层。剪映草稿包含底片、窗格与踏板、光束与亮边三轨。

## 安装与使用

需要 Python 3.10+，以及可从命令行调用的 FFmpeg 和 ffprobe。FFmpeg 需包含 `libx264` 和 `prores_ks` 编码器。生成剪映草稿还需要剪映专业版和已安装的 `jianying-editor` 技能。仅渲染 MP4 不需要安装剪映。

以下是 Windows PowerShell 示例。

```powershell
$codexDir = if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $HOME '.codex' }
$skills = Join-Path $codexDir 'skills'
New-Item -ItemType Directory -Path $skills -Force | Out-Null
$skill = Join-Path $skills 'window-tunnel-opening'
git clone https://github.com/zhengtangbo/window-tunnel-opening.git $skill
python -m pip install -r "$skill/requirements.txt"
```

在 Codex 中选中 **窗格隧道打开**（`$window-tunnel-opening`），上传视频，发送「做同款开场」。

### 从命令行生成

```powershell
python "$skill/scripts/make_opening.py" "D:/media/travel.mp4" `
  --work-dir "./work/opening" --output "./outputs/opening.mp4"
```

使用新建或空的工作目录。用 `--start 2 --seconds 8` 从第 2 秒取 8 秒，用 `--mute` 去掉原声。需要换踏板纹理时，用 `--texture 0.4 0.82` 从原片指定位置取样。坐标范围为 0 到 1，原点位于左上角。

自动识别成功后，工作目录里的 `road-check.jpg` 会标出开头、中段以及揭开时的道路边线与中心线。岔路、遮挡和明显转镜可能需要校正。静态镜头可追加例如 `--road-point 0.30 0.98 0.32 --road-point 0.47 0.76 0.16 --road-point 0.58 0.52 0.05`，每组依次为中心横坐标、纵坐标和道路全宽。移动镜头使用 `--road-track` 传入[路线关键帧](references/road-track.md)。重新渲染时指定新的工作目录和输出文件。用 `--no-road` 可选择通用隧道动线。

### 生成剪映草稿

先运行上一条渲染命令，再把 `$draftRoot` 设为剪映专业版设置中的本地草稿位置。脚本会创建一个新的自包含草稿。

```powershell
python "$skill/scripts/make_draft.py" --work-dir "./work/opening" `
  --draft-root "$draftRoot" --name "旅行窗格开场"
```

脚本会在个人技能目录或同级目录查找 `jianying-editor`。如果安装在其他位置，可把 `JY_SKILL_ROOT` 设为它的技能根目录。透明层使用预乘 Alpha 的 ProRes 4444，窗格形状及扫开动效在脚本中生成，剪映中可以分别调整两层动效的时间与不透明度。

## 许可证

本仓库代码和文档采用 [MIT 许可证](LICENSE)。输入素材由使用者提供。
