import torch
from sg_utils.sg_util import get_metrics
import os
from main import parse_arguments
from model.SG_Transformer import TransformerSG
from sg_utils.sg_util import load_data_from_npy, test_model
from data_preprocess.dataset import SceneGraphDataset, DataLoader, collate_fn


def main(args):
    save_model_path = os.path.join(f"{args.save_model_path}", f"{args.dataset}")
    # 获取保存的模型文件列表
    model_files = [f for f in os.listdir(save_model_path) if f.endswith('.pth')]

    #获取test数据
    dataset_dir = os.path.join(f"{args.data_path}", f"{args.dataset}")
    test_file_path = os.path.join(dataset_dir, "embedding", "val_embeddings.npy")
    test_data = load_data_from_npy(test_file_path)
    test_dataset = SceneGraphDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerSG(args.feature_dim, args.resnet_dim, args.n_heads, args.num_encoder_layers,
                          args.dim_feedforward, args.num_classes, args.d_k, args.d_v)
    for model_file in model_files:
        print(f"Evaluating model: {model_file}")
        model.load_state_dict(torch.load(os.path.join(save_model_path, model_file)))
        model = model.to(device)

        # 测试模型并打印评估指标
        print(f"Results for {model_file}:")
        avg_loss, acc, pub_prec, pub_rec, priv_prec, priv_rec, cm, macro_f1 = test_model(model, test_loader, device)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
    # import requests
    #
    # try:
    #     response = requests.get("https://live.staticflickr.com/2704/4381133464_07596ce22f_o.jpg")
    #     if response.status_code == 200:
    #         print("true")
    #     else:
    #         print("else false")
    # except requests.exceptions.RequestException:
    #     print("false")