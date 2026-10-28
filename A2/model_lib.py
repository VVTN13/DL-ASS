"""Five-stage EEG classifiers for waveform batches shaped (B, channels, 30, 200)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CBraModClassifier(nn.Module):
    """Pretrained CBraMod for five-stage Sleep-EDF waveform classification."""

    def __init__(self, in_channels=2, checkpoint_path="pretrained_weights/pretrained_weights.pth",
                 freeze_encoder=False, num_classes=5, dropout=0.5):
        super().__init__()
        if __package__:
            from .cbramod import CBraMod
        else:
            from cbramod import CBraMod

        self.encoder = CBraMod()
        weights = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.encoder.load_state_dict(weights, strict=True)
        self.encoder.proj_out = nn.Identity()
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(in_channels * 30 * 200, 800), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(800, 200), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(200, num_classes),
        )

    def forward(self, x):
        # Keep the 200-point FFT in float32 while the encoder can use CUDA AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            embeddings = self.encoder.patch_embedding(x.float())
        return self.classifier(self.encoder.encoder(embeddings))


# Model
class ConvEmbedding(nn.Module):
    """Encode each EEG channel's waveform grid as one transformer token."""
    def __init__(self, in_channels, embed_dim, kernel_size=3, stride=1, padding=1):
        super(ConvEmbedding, self).__init__()
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2d(1, 8, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)  # ReLU activation after batch normalization
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout1 = nn.Dropout(p=0.25)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.relu2 = nn.ReLU(inplace=True)  # ReLU activation after batch normalization
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.dropout3 = nn.Dropout(p=0.25)
        
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(8)
        self.relu4 = nn.ReLU(inplace=True)  # ReLU activation after batch normalization
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.dropout5 = nn.Dropout(p=0.25)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc9 = nn.Linear(8, embed_dim)
        self.relu9 = nn.ReLU(inplace=True) 

        # Class token and positional embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(torch.randn(1, in_channels + 1, embed_dim))

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B * C, 1, H, W)
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))  # conv1 + batchnorm1 + relu1 + pooling1
        x = self.dropout1(x)
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))  # conv2 + batchnorm2 + relu2 + pooling2
        x = self.dropout3(x)  # dropout3
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))  # conv4 + batchnorm4 + relu4 + pooling4      
        x = self.dropout5(x)
        x = self.gap(x).flatten(1)
        x = self.relu9(self.fc9(x))  # fc9 + relu
        x = x.reshape(B, C, self.embed_dim)

        # Add cls_token to the sequence
        cls_tokens = self.cls_token.expand(B, -1, -1)  # Shape: (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: (B, C+1, embed_dim)

        # Add position embedding
        x = x + self.position_embedding
        return x  # Shape: (B, C+1, embed_dim)

class ConvLSTMAttentionEmbedding(nn.Module):
    """Encode each EEG channel with CNN, temporal LSTM and attention pooling."""
    def __init__(self, in_channels, embed_dim, lstm_hidden_size=128):
        super(ConvLSTMAttentionEmbedding, self).__init__()
        self.embed_dim = embed_dim

        # --- CNN Feature Extractor ---
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True) 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.25)
        
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.relu2 = nn.ReLU(inplace=True) 
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(p=0.25)
        
        # --- Temporal LSTM Block ---
        self.lstm = nn.LSTM(
            input_size=8,
            hidden_size=lstm_hidden_size, 
            num_layers=1, 
            batch_first=True
        )
        
        # --- NEW: Temporal Attention Network ---
        # This calculates an "importance score" for each time step
        self.temporal_attention = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.Tanh(),
            nn.Linear(lstm_hidden_size // 2, 1)
        )
        
        # Final projection
        self.fc9 = nn.Linear(lstm_hidden_size, embed_dim)
        self.relu9 = nn.ReLU(inplace=True) 

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(torch.randn(1, in_channels + 1, embed_dim))

    def forward(self, x):
        B, C, seconds, samples_per_second = x.shape
        x = x.reshape(B * C, 1, seconds, samples_per_second)
        
        # 1. Extract features from each channel's waveform grid.
        x = self.pool1(self.relu1(self.bn1(self.conv1(x)))) 
        x = self.dropout1(x)
        x = self.pool2(self.relu2(self.bn2(self.conv2(x)))) 
        x = self.dropout3(x)
        # 2. Flatten the pooled time grid in row-major order, retaining 8 features.
        # (B*C, 8, H, W) -> (B*C, H*W, 8), independent of the grid dimensions.
        x = x.flatten(2).transpose(1, 2).contiguous()
        
        # 3. Temporal Modeling
        lstm_out, _ = self.lstm(x)  # (B*C, pooled time steps, lstm_hidden_size)
        
        # --- 4. NEW: Condense Sequence with Attention ---
        attn_scores = self.temporal_attention(lstm_out)
        
        # Normalize scores so they sum to 1.0 (creates a probability distribution over time)
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # Multiply each time step by its weight, then sum them all together
        x = torch.sum(lstm_out * attn_weights, dim=1) 
        # Shape: (B*C, lstm_hidden_size)
        
        # 5. Final Projection & Transformer Setup
        x = self.relu9(self.fc9(x)) 
        x = x.reshape(B, C, self.embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, x), dim=1) 

        x = x + self.position_embedding 
        
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_dim=768, num_heads=12, dropout=0.25):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (emb_dim // num_heads) ** -0.5
        self.qkv = nn.Linear(emb_dim, emb_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, record_attn=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        if record_attn:
            # Fall back to manual math if you explicitly need to save the attention matrix
            attn = (q @ k.transpose(-2, -1)) * self.scale
            r_attn = attn
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        else:
            # FAST PATH: Use PyTorch SDPA 
            # (Make sure to pass dropout probability only during training)
            dropout_p = self.attn_drop.p if self.training else 0.0
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
            x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        if record_attn:
            return x, r_attn
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, emb_dim=768, num_heads=16, mlp_ratio=4, dropout=0.3):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = MultiHeadSelfAttention(emb_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_ratio * emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * emb_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, record_attn=False):
        if record_attn:
            attn_output, attn_matrix = self.attn(self.norm1(x), record_attn=True)
            x = x + attn_output
            x = x + self.mlp(self.norm2(x))
            return x, attn_matrix
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x

class ConvLSTMAttentionTransformer(nn.Module):
    def __init__(self, img_size: int = 112, in_channels: int = 2, emb_dim: int = 256,
                 num_heads: int = 8, num_layers: int = 4, mlp_ratio: int = 3, num_classes: int = 5,
                 lstm_hidden_size: int = 128):
        super().__init__()
        # img_size is retained for earlier calls; grid dimensions are inferred at forward.
        self.patch_embed = ConvLSTMAttentionEmbedding(
            in_channels=in_channels, embed_dim=emb_dim, lstm_hidden_size=lstm_hidden_size
        )
        self.encoder = nn.ModuleList(
            [TransformerEncoderLayer(emb_dim, num_heads, mlp_ratio) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, num_classes)

    def forward(self, x, record_attn=False):
        x = self.patch_embed(x)
        attn_matrices = [] if record_attn else None

        for layer in self.encoder:
            if record_attn:
                x, attn_matrix = layer(x, record_attn=True)
                attn_matrices.append(attn_matrix)
            else:
                x = layer(x)

        x = self.norm(x[:, 0])  # Only the CLS token
        x = self.head(x)

        # Return both output and attention matrices if recording
        if record_attn:
            return x, attn_matrices
        return x
    
class Transformer(nn.Module):
    def __init__(self, img_size: int = 112, in_channels: int = 2, emb_dim: int = 256,
                 num_heads: int = 8, num_layers: int = 4, mlp_ratio: int = 3, num_classes: int = 5):
        super().__init__()
        # img_size is retained for earlier calls; adaptive pooling handles the grid size.
        self.patch_embed = ConvEmbedding(in_channels=in_channels, embed_dim=emb_dim)
        self.encoder = nn.ModuleList(
            [TransformerEncoderLayer(emb_dim, num_heads, mlp_ratio) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, num_classes)

    def forward(self, x, record_attn=False):
        x = self.patch_embed(x)
        attn_matrices = [] if record_attn else None

        for layer in self.encoder:
            if record_attn:
                x, attn_matrix = layer(x, record_attn=True)
                attn_matrices.append(attn_matrix)
            else:
                x = layer(x)

        x = self.norm(x[:, 0])  # Only the CLS token
        x = self.head(x)

        # Return both output and attention matrices if recording
        if record_attn:
            return x, attn_matrices
        return x
    
class StandardCNN2D(nn.Module):
    """CNN baseline for (batch, EEG channels, seconds, samples/second) waveforms."""

    def __init__(self, in_channels=2, num_classes=5):
        super(StandardCNN2D, self).__init__()
        
        self.features = nn.Sequential(
            # Input shape for Sleep-EDF: (Batch_Size, 2, 30, 200)
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            # Output shape: (Batch_Size, 32, 15, 100)
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            # Output shape: (Batch_Size, 64, 7, 50)
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            # Output shape: (Batch_Size, 128, 3, 25)
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  
            # Output shape: (Batch_Size, 256, 1, 12)
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
            # Output shape: (Batch_Size, num_classes) - raw logits for CrossEntropyLoss
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        # MLP shared between both poolings
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, max(1, in_planes // ratio), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, in_planes // ratio), in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pool across the channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        # First refine features by channel, then refine spatially
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class CBAM_EEG_Net(nn.Module):
    def __init__(self, in_channels=2, num_classes=5):
        super(CBAM_EEG_Net, self).__init__()

        # --- Block 1 ---
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.cbam1 = CBAM(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(p=0.2) # Spatial Dropout

        # --- Block 2 ---
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.cbam2 = CBAM(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(p=0.25)

        # --- Block 3 ---
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.cbam3 = CBAM(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout2d(p=0.3)
        
        # --- Block 4 ---
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)
        self.cbam4 = CBAM(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout2d(p=0.3)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flattened_size = 256

        # Classifier
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.relu_fc = nn.ReLU(inplace=True)
        self.drop_fc = nn.Dropout(p=0.6) 
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Sleep-EDF waveform input: (Batch, 2, 30, 200).
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.cbam1(x)
        x = self.pool1(x)
        x = self.drop1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.cbam2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.cbam3(x)
        x = self.pool3(x)
        x = self.drop3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.cbam4(x)
        x = self.pool4(x)
        x = self.drop4(x)

        x = self.gap(x).flatten(1)

        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.drop_fc(x)
        x = self.fc2(x)
        
        return x
    
class ResidualCBAMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p):
        super(ResidualCBAMBlock, self).__init__()
        
        # Main Path
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.cbam = CBAM(out_channels)
        
        # Shortcut Path (Residual Connection)
        # If the number of channels changes, we use a 1x1 conv to match them up
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity() # Just pass it through if shapes match
            
        # Post-addition operations
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        # Save the shortcut
        residual = self.shortcut(x)
        
        # Forward the main path
        out = self.conv(x)
        out = self.bn(out)
        out = self.cbam(out)
        
        # Add the residual BEFORE the activation function (Standard ResNet design)
        out += residual
        out = self.relu(out)
        
        # Pool and Drop
        out = self.pool(out)
        out = self.drop(out)
        return out

class ResCBAM_EEG_Net(nn.Module):
    def __init__(self, in_channels=2, num_classes=5):
        super(ResCBAM_EEG_Net, self).__init__()

        # Stack the modular residual blocks
        self.layer1 = ResidualCBAMBlock(in_channels, 32, dropout_p=0.2)
        self.layer2 = ResidualCBAMBlock(32, 64, dropout_p=0.25)
        self.layer3 = ResidualCBAMBlock(64, 128, dropout_p=0.3)
        self.layer4 = ResidualCBAMBlock(128, 256, dropout_p=0.3)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.fc1 = nn.Linear(256, 64)
        self.relu_fc = nn.ReLU(inplace=True)
        self.drop_fc = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Sleep-EDF waveform input: (Batch, 2, 30, 200).
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Condense spatial dimensions to 1x1
        x = self.gap(x)
        x = x.flatten(1)

        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.drop_fc(x)
        x = self.fc2(x)
        
        return x
    
# ==========================================
# 2. ResNet50 Bottleneck Block + CBAM
# ==========================================
class CBAMBottleneck(nn.Module):
    # ResNet50 expands the channels by 4x at the end of every block
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(CBAMBottleneck, self).__init__()
        
        # 1x1 Convolution (Reduces channels)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # 3x3 Convolution (Main feature extraction)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # 1x1 Convolution (Expands channels back up)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        
        # --- CBAM applied to the expanded channels ---
        self.cbam = CBAM(planes * self.expansion, ratio=16)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 1. Apply CBAM attention to the extracted features
        out = self.cbam(out)

        # 2. Project the shortcut (if dimensions changed)
        if self.downsample is not None:
            identity = self.downsample(x)

        # 3. Add the shortcut BEFORE the final ReLU
        out += identity
        out = self.relu(out)

        return out

# ==========================================
# 3. Main ResNet50 + CBAM Architecture
# ==========================================
class ResNet50_CBAM(nn.Module):
    def __init__(self, in_channels=2, num_classes=5):
        super(ResNet50_CBAM, self).__init__()
        self.inplanes = 64

        # --- Stem: Initial Convolution ---
        # Match the number of EEG channels in the dataset.
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # --- ResNet50 Layers (Bottleneck configurations: [3, 4, 6, 3]) ---
        self.layer1 = self._make_layer(CBAMBottleneck, 64,  3, stride=1)
        self.layer2 = self._make_layer(CBAMBottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(CBAMBottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(CBAMBottleneck, 512, 3, stride=2)

        # --- Classifier ---
        # ResNet50 ends with 2048 channels (512 * 4 expansion)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512 * CBAMBottleneck.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        
        # If the stride is not 1, or the input channels don't match the output,
        # we need a 1x1 convolution on the shortcut to align the shapes.
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # First block handles the downsampling/stride
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        
        # Remaining blocks maintain the same dimension
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Sleep-EDF waveform input: (Batch, 2, 30, 200).
        
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual Layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Pooling and Classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
