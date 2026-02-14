import torch

@torch.no_grad()
def _ema_update(student_model, teacher_model, momentum=0.996):
    for student_params, teacher_params in zip(student_model.parameters(), teacher_model.parameters()):
        teacher_params.data.mul_(momentum).add_(student_params.data, alpha=1.0-momentum)
