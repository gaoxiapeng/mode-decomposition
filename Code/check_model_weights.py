import torch
import sys

def check_model_weights(checkpoint_path):
    """检查模型权重是否包含NaN"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        print(f"Checkpoint keys: {ckpt.keys()}")
        print(f"Epoch: {ckpt.get('epoch', 'N/A')}")
        print(f"Loss: {ckpt.get('loss', 'N/A')}")
        
        state_dict = ckpt['model_state_dict']
        print(f"\nModel has {len(state_dict)} parameters")
        
        nan_found = False
        inf_found = False
        
        for key, value in state_dict.items():
            if torch.isnan(value).any():
                print(f"  ❌ NaN found in {key}: shape={value.shape}, count={torch.isnan(value).sum()}")
                nan_found = True
            
            if torch.isinf(value).any():
                print(f"  ⚠️  Inf found in {key}: shape={value.shape}, count={torch.isinf(value).sum()}")
                inf_found = True
            
            # 检查异常大的值
            if torch.abs(value).max() > 1e6:
                print(f"  ⚠️  Large values in {key}: max={torch.abs(value).max():.2e}")
        
        if not nan_found and not inf_found:
            print("\n✅ No NaN or Inf found in model weights!")
        else:
            print("\n❌ Model weights contain NaN or Inf - need to retrain!")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

if __name__ == '__main__':
    checkpoint_path = '../exp/vmd/best_model.pth.tar'
    check_model_weights(checkpoint_path)