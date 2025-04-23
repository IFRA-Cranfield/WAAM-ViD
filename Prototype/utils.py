import torch

def testing_loop(model, loader, device=torch.device('cuda')):
    predict_width_list = []
    
    with torch.no_grad():
        for batch in loader:
            # extract inputs and labels from the batch
            input1 = batch["predict_mask"].to(device, dtype=torch.float32)
            input1 = input1.repeat(1, 3, 1, 1)  # repeat input1 to match the expected input shape of the model

            input2 = torch.cat([
                batch["camera_matrix"].to(device, dtype=torch.float32),
                batch["distortion_coefficients"].to(device, dtype=torch.float32),
                batch["rotation_matrix"].to(device, dtype=torch.float32),
                batch["translation_vector"].to(device, dtype=torch.float32)
            ], dim=1)  # concatenate input2 features

            # forward pass
            outputs = model(input1, input2)

            # move outputs and labels to CPU
            outputs = outputs.cpu()

            for predict_width in outputs.numpy():
                predict_width_list.append(predict_width)

    return predict_width_list