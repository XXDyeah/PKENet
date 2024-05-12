import torch
import numpy as np
import matplotlib.pyplot as plt

def compute_features_FeatCluster_new(ref_model, tg_model, dataloader, num_samples, num_features, device):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ref_model.eval()
    tg_model.eval()

    inputs_all = torch.zeros([num_samples, 3, 32, 32])
    targets_all = np.zeros([num_samples]).astype(np.int32)
    old_features = np.zeros([num_samples, num_features])
    new_features = np.zeros([num_samples, num_features])
    memory_flags_all = np.zeros([num_samples]).astype(np.float64)

    start_idx = 0
    with torch.no_grad():
        for input_dict in dataloader:
            inputs = input_dict["inputs"]
            if not isinstance(inputs, torch.Tensor):
                raise ValueError("Inputs must be a torch.Tensor")
            inputs = inputs.to(device)

            # features_old_tensor = ref_model(inputs)['features']
            features_old_tensor = ref_model(inputs)['raw_features']
            old_features[start_idx:start_idx + inputs.shape[0], :] = np.squeeze(features_old_tensor.cpu())
            # features_new_tensor = tg_model(inputs)['features']
            features_new_tensor = tg_model(inputs)['raw_features']
            new_features[start_idx:start_idx + inputs.shape[0], :] = np.squeeze(features_new_tensor.cpu())
            inputs_all[start_idx:start_idx + inputs.shape[0], :, :, :] = inputs
            targets_all[start_idx:start_idx + inputs.shape[0]] = input_dict["targets"]
            memory_flags_all[start_idx:start_idx + inputs.shape[0]] = input_dict["memory_flags"]

            start_idx = start_idx + inputs.shape[0]

    assert (start_idx == num_samples)
    return inputs_all, old_features, new_features, targets_all, memory_flags_all


def compute_features_FeatCluster(ref_feature_model, tg_feature_model, dataloader, num_samples, num_features, device):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ref_feature_model.eval()
    tg_feature_model.eval()

    inputs_all = torch.zeros([num_samples, 3, 32, 32])
    targets_all = np.zeros([num_samples]).astype(np.int32)
    old_features = np.zeros([num_samples, num_features])
    new_features = np.zeros([num_samples, num_features])
    memory_flags_all = np.zeros([num_samples]).astype(np.float64)

    start_idx = 0
    with torch.no_grad():
        for input_dict in dataloader:
            inputs = input_dict["inputs"]
            if not isinstance(inputs, torch.Tensor):
                raise ValueError("Inputs must be a torch.Tensor")
            inputs = inputs.to(device)
            # output = ref_feature_model(inputs)
            # print(type(output))
            # print(output)

            features_old_tensor = ref_feature_model(inputs)['raw_features']
            old_features[start_idx:start_idx + inputs.shape[0], :] = np.squeeze(features_old_tensor.cpu())
            features_new_tensor = tg_feature_model(inputs)['raw_features']
            new_features[start_idx:start_idx + inputs.shape[0], :] = np.squeeze(features_new_tensor.cpu())
            inputs_all[start_idx:start_idx + inputs.shape[0], :, :, :] = inputs
            targets_all[start_idx:start_idx + inputs.shape[0]] = input_dict["targets"]
            memory_flags_all[start_idx:start_idx + inputs.shape[0]] = input_dict["memory_flags"]

            # for i in range(inputs_all.size(0)):
            #     each_imgae = inputs_all[i,...].cpu().numpy()
            #     print(targets_all[i])
            #     each_imgae = each_imgae.transpose(1,2,0)
            #     each_imgae = each_imgae.copy()
            #     each_imgae = (((each_imgae-each_imgae.min())/(each_imgae.max()-each_imgae.min()))*255).astype(np.uint8)
            #     plt.imshow(each_imgae)
            #     plt.savefig("{}.png".format(i))

            start_idx = start_idx + inputs.shape[0]

    assert (start_idx == num_samples)
    # if not return_data:
    #     return old_features
    # else:
    #     return inputs_all, old_features, new_features, targets_all
    return inputs_all, old_features, new_features, targets_all, memory_flags_all


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)
    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res