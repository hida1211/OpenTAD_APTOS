# 独自データセットでの ActionFormer 利用手順

以下の手順では、OpenTAD に含まれる ActionFormer を使用し、独自のデータセットをトレーニング・検証、テストデータで推論を行う方法をまとめます。

## 1. 環境構築

1. Google Colab の GPU ランタイムを使用していることを確認します。
2. Python と PyTorch をインストールします。推奨バージョンは Python 3.10.12、PyTorch 2.0.1 です。
   Colab で `!nvidia-smi` を実行し、表示される CUDA バージョンに合わせて `pytorch-cuda=<version>` を変更してください。
3. MMAaction2 を導入します。
4. 本リポジトリをクローンし、依存ライブラリをインストールします。

```bash
conda create -n opentad python=3.10.12
conda activate opentad
# ここでは CUDA 11.8 を例としているため、表示されたバージョンに合わせて変更します
conda install pytorch=2.0.1 torchvision=0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install openmim
mim install mmcv==2.0.1
mim install mmaction2==1.1.0
# リポジトリを取得
git clone https://github.com/sming256/OpenTAD.git
cd OpenTAD
pip install -r requirements.txt
```

## 2. データセットの準備

1. `data/<dataset_name>/annotations/` にアノテーション JSON を配置します。フォーマットは以下の例を参照してください。

```json
{
  "version": "1.0",
  "database": {
    "video_0001": {
      "subset": "train",
      "duration": 120.0,
      "frame": 3600,
      "annotations": [
        {"label": "action_a", "segment": [1.2, 3.4]},
        {"label": "action_b", "segment": [10.0, 15.0]}
      ]
    },
    "video_0002": {
      "subset": "val",
      "duration": 90.0,
      "frame": 2700,
      "annotations": [
        {"label": "action_a", "segment": [5.0, 7.0]}
      ]
    }
  }
}
```

2. クラス名一覧を `category_idx.txt` として保存します（1 行に 1 クラス）。
3. 事前に抽出した特徴量を `data/<dataset_name>/features/` に配置します。特徴量が無い場合は生動画を `data/<dataset_name>/raw_data/` に置き、End-to-End 学習用に設定します。
4. 欠損している特徴量がある場合は `missing_files.txt` を作成し、`data/<dataset_name>/features/` に保存します。`python tools/prepare_data/generate_missing_list.py <annotation.json> <feature_dir>` で生成できます。

## 3. 設定ファイルの作成

`configs/_base_/datasets/` 以下の既存ファイルを参考に、独自データセット用の設定ファイルを作成します。主な項目は次の通りです。

```python
dataset_type = "ThumosPaddingDataset"  # データ形式に応じて変更
annotation_path = "data/<dataset_name>/annotations/your_annotation.json"
class_map = "data/<dataset_name>/annotations/category_idx.txt"
data_path = "data/<dataset_name>/features/"
block_list = data_path + "missing_files.txt"
```

この設定ファイルを `configs/actionformer/your_dataset.py` として保存し、モデル設定 `_base_/models/actionformer.py` を読み込みます。

## 4. トレーニング

次のコマンドでトレーニングを実行します。`--nproc_per_node` は使用 GPU 数です。

```bash
torchrun --nnodes=1 --nproc_per_node=1 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    tools/train.py configs/actionformer/your_dataset.py
```

訓練中に自動で検証が実行され、結果が `work_dir` に保存されます。

## 5. 検証データでの評価

学習済み重みを指定して検証を実行します。

```bash
torchrun --nnodes=1 --nproc_per_node=1 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    tools/test.py configs/actionformer/your_dataset.py \
    --checkpoint <path_to_checkpoint>
```

`evaluation` の設定に従い、mAP などの指標が計算されます。

## 6. テストデータでの推論

テスト用アノテーションを `subset: "test"` として準備し、次のコマンドで予測結果を取得します。

```bash
torchrun --nnodes=1 --nproc_per_node=1 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    tools/test.py configs/actionformer/your_dataset.py \
    --checkpoint <path_to_checkpoint> \
    --out result.json
```

`result.json` に各動画の推論結果が保存されます。`inference.load_from_raw_predictions` を `True` にすると、保存済みの生の予測を再利用できます。

