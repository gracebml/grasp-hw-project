# Adapted from LGD, Copyright (c) 2024 An Vuong, MIT License (THIRD_PARTY_NOTICES.md).
import torch.nn.functional as F

from . import refs

refs.add_lgd_to_path()
from inference.models.lgrconvnet3 import GenerativeResnet  # noqa: E402
from inference.models.lggcnn import LGGCNN  # noqa: E402


class GenerativeResnetNoDetach(GenerativeResnet):
    def forward(self, x_in, prompt, query):
        x = F.relu(self.bn1(self.conv1(x_in)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)

        device = x.device
        y_feats = self._encode_text(query, device=device)
        y_feats = self.y_flatten(y_feats)
        y_feats = y_feats.unsqueeze(2).expand(-1, -1, 56).unsqueeze(1).expand(-1, 128, -1, -1)

        x = x + y_feats  # the official code detaches x here

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)

        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x))
            cos_output = self.cos_output(self.dropout_cos(x))
            sin_output = self.sin_output(self.dropout_sin(x))
            width_output = self.width_output(self.dropout_wid(x))
        else:
            pos_output = self.pos_output(x)
            cos_output = self.cos_output(x)
            sin_output = self.sin_output(x)
            width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output


class LGGCNNNoDetach(LGGCNN):
    def forward(self, x, prompt, query):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        device = x.device
        y_feats = self._encode_text(query, device=device)
        y_feats = self.y_flatten(y_feats)
        y_feats = y_feats.unsqueeze(2).expand(-1, -1, 19).unsqueeze(1).expand(-1, 8, -1, -1)

        x = x + y_feats  # the official code detaches x here

        x = F.relu(self.convt1(x))
        x = F.relu(self.convt2(x))
        x = F.relu(self.convt3(x))

        pos_output = self.pos_output(x)
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output
