import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration

class PoseT5(nn.Module):
    def __init__(self, model_name="t5-small"):
        super().__init__()

        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)

        d_model = self.t5.config.d_model

        # 🔥 FLOAT → d_model projection (IMPORTANT)
        self.input_projection = nn.Linear(1, d_model)

    def forward(self, pose, input_ids, attention_mask):
        # pose: (batch_size, 66)

        # reshape → (batch_size, 66, 1)
        pose = pose.unsqueeze(-1)

        # project → (batch_size, 66, d_model)
        pose_embeds = self.input_projection(pose)

        # pass as encoder input
        outputs = self.t5(
            inputs_embeds=pose_embeds,
            attention_mask=torch.ones(pose_embeds.size()[:2]).to(pose.device),
            labels=input_ids
        )

        return outputs