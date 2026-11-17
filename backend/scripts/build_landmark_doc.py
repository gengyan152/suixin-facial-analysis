"""Generate the exhaustive index appendix from installed MediaPipe topology."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.geometry import REGIONS,SYMMETRY_PAIRS
from app.config import ROOT
import mediapipe as mp


def main():
    connections=mp.tasks.vision.FaceLandmarksConnections
    edges=[(c.start,c.end) for c in connections.FACE_LANDMARKS_TESSELATION]
    names={0:"上唇外缘中央",1:"鼻尖",2:"鼻下点代理",9:"眉心代理（中庭上端）",10:"前额网格上缘（不是发际线）",
           13:"上唇内缘中央",14:"下唇内缘中央",17:"下唇外缘中央",33:"图像左眼外眦",133:"图像左眼内眦",
           263:"图像右眼外眦",362:"图像右眼内眦",61:"图像左嘴角",291:"图像右嘴角",98:"图像左鼻翼代理",
           327:"图像右鼻翼代理",152:"颏底",168:"鼻根（非眉心）",6:"鼻梁中轴",468:"图像左虹膜中心",473:"图像右虹膜中心"}
    for i in range(469,473): names[i]="图像左虹膜环"
    for i in range(474,478): names[i]="图像右虹膜环"
    text='''# MediaPipe 478 点索引与测量契约

本表对应后端 `app/geometry.py`，采用 0 起始索引。0—467 为标准脸部网格，468—477 为虹膜扩展。
“图像左/右”指 EXIF 转正后的未镜像像素图；未镜像正面图中分别对应人物自身右/左。客户端镜像预览不改变服务端索引。
坐标先反归一化为 `x*width, y*height`；裁剪预测再加裁剪左上角偏移。不能直接在归一化坐标上量非正方形图像距离。

## 解剖标志与传统索引

68 点列使用 dlib/iBUG 的 0 起始约定；列出解剖语义对应而非宣称两个模型顶点重合。106 点没有跨库统一编号：InsightFace、PFLD、厂商 SDK 的定义可能不同，当前没有指定 106 点模型，因此不虚构唯一数字表。106 点接入时须声明模型及版本，再按本表的解剖语义匹配并做人工核验。

| 标志 | MediaPipe 478 | iBUG 68 | 106 点对接语义 |
|---|---|---|---|
| 眉心 | 9（代理） | 21、22 中点（代理） | 双眉内端中点 |
| 鼻根 | 168 | 27 | 鼻梁上端 |
| 图像左外/内眦 | 33 / 133 | 36 / 39 | 左眼外/内眼角 |
| 图像右内/外眦 | 362 / 263 | 42 / 45 | 右眼内/外眼角 |
| 图像左/右眼上下睑中心 | 159、145 / 386、374 | 37/38、40/41 均值；43/44、46/47 均值 | 上下睑中点 |
| 图像左/右眉下缘 | 52、65 / 282、295 | 19/20、23/24 附近，无独立上下眉缘 | 眉下缘 |
| 鼻尖 | 1 | 30 | 鼻尖 |
| 左/右鼻翼 | 98 / 327（代理） | 31 / 35 | 鼻翼最外缘 |
| 鼻下点 | 2（代理） | 33（代理） | 鼻小柱下缘 |
| 左/右唇峰 | 37 / 267 | 50 / 52 | 上唇双峰 |
| 上唇外/内中央 | 0 / 13 | 51 / 62 | 上唇厚度端点 |
| 下唇内/外中央 | 14 / 17 | 66 / 57 | 下唇厚度端点 |
| 左/右嘴角 | 61 / 291 | 48 / 54 | 嘴角 |
| 颏底 | 152 | 8 | 下颌最低点 |
| 左颊/右颊 | 234 / 454 | 1/15 附近 | 外侧轮廓 |
| 下颌弧（图像左到右） | 234,93,132,58,172,136,150,149,176,148,152,377,400,378,379,365,397,288,361,323,454 | 0—16（采样密度不同） | 下颌轮廓序列 |

## 测量定义

1. 眼中心由外眦、内眦平均得到，以二者连线旋转到水平；所有下面的距离均在同一像素平面计算。
2. 五眼：眼裂宽、内眦距除以眼中心水平线与完整脸部椭圆交点之间的面宽；`inner_canthus_to_eye_width_ratio` 为内眦距除以双眼平均眼裂宽。
3. 眉眼比：左右分别计算眉下缘均值至眼中心的垂直距离 / 眼裂高度，再取两侧平均。
4. 中下庭：`(y[2]-y[9])/(y[152]-y[2])`；没有真实发际线，因此不输出上庭。
5. 鼻唇：`abs(x[327]-x[98])/(x[362]-x[133])`；上下唇比 `(y[13]-y[0])/(y[17]-y[14])`。
6. 镜像：对 168、6、2、152 做 PCA 直线拟合；镜像反射左侧成对点，计算与右侧点的二维距离均值，除以双外眦距得 `d`；显示对称度 `max(0,1-2d)`。这是几何显示变换，没有人口常模阈值。
7. 姿态：从 MediaPipe 4×4 矩阵的左上 3×3 经 SVD 去尺度，按 OpenCV RQ 分解提取 pitch/yaw/roll（度），不是像素几何的二维眼线 Roll。

缺测判定：不是 478 点、非有限数值、轮廓面积小于 100 px²、眼/嘴左右顺序翻转、分母非正或退化、眼水平线不与轮廓相交，均停止数值输出。姿态/清晰度/亮度检查只作用于选中的人脸。478 网格不提供可靠的逐点遮挡概率；遮挡由后置描述标为无法判断。

训练侧 `scut-geometry-v1` 的眉心、眉下缘、镜像中轴与成对点已只读核对；当前后端眉眼比是左右统计字段的均值。

## 分区与局部模糊

区域优先级为眼、眉、鼻、嘴、轮廓，先占像素排除后续重叠。眼与眉分别构造双侧掩码，轮廓只保留外缘带。每一区域固定 3 个同二值面积平移参照，原图 + 5 区域 + 15 参照共 21 张一次前向；边界高斯羽化 15×15、sigma=4，图像模糊 31×31、sigma=9。参照是裁剪内随机位置；不能作为显著性检验。`is_sensitive` 是相对参照绝对差大于 0.05 的展示阈值。

'''
    for key,ids in REGIONS.items(): text+=f"- `{key}`: `{','.join(map(str,ids))}`\n"
    text+=f"\n镜像配对：`{SYMMETRY_PAIRS}`\n"
    text+='''
## 全部 478 点编号登记

下表逐点登记语义或网格身份、使用分区及直接相邻索引。官方只为部分解剖环提供命名；“面部网格顶点”明确表示没有单独解剖名称，不能擅自给每个顶点命名为传统标志。
邻接来自安装版本 MediaPipe 的 `FACE_LANDMARKS_TESSELATION`；虹膜环补充其明确的四边连接，中心不属于三角网格。

| 索引 | 解剖语义/网格身份 | 测量分区 | 拓扑邻点 |
|---:|---|---|---|
'''
    iris_edges=[(469,470),(470,471),(471,472),(472,469),(474,475),(475,476),(476,477),(477,474)]
    for i in range(478):
        adjacent=sorted({b if a==i else a for a,b in edges+iris_edges if a==i or b==i})
        regions=','.join(key for key,ids in REGIONS.items() if i in ids) or ('iris' if i>=468 else 'mesh')
        text+=f"| {i} | {names.get(i,'面部网格顶点')} | {regions} | {','.join(map(str,adjacent)) or '中心独立点'} |\n"
    text+='''
## 官方来源

- [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python)
- [MediaPipe 官方连接定义](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/vision/face_landmarker.py)
- [MediaPipe 标准脸部拓扑模型](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj)

版本及二进制 SHA-256 见 `backend/requirements.lock.txt` 与 `backend/weights/manifest.json`；执行 `python scripts/build_landmark_doc.py` 可按当前固定代码与拓扑重新生成。
'''
    (ROOT.parent/'docs/LANDMARK_MAPPING.md').write_text(text,encoding='utf-8')


if __name__=='__main__': main()
