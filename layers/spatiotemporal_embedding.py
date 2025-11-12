import torch
import torch.nn as nn


class SpatioTemporalEmbedding(nn.Module):
    """
    Spatiotemporal embedding module that encodes temporal patterns.
    Combines hour-of-day and day-of-week embeddings for urban flow data.
    """

    def __init__(self, time, features, dropout=0.1):
        super(SpatioTemporalEmbedding, self).__init__()

        self.time = time
        # Learnable embeddings for hour of day
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)

        # Learnable embeddings for day of week (7 days)
        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        """
        Args:
            x: Input tensor with temporal information
               Shape: (batch_size, seq_len, num_nodes, 3)
               where last dimension contains [flow_value, hour_of_day, day_of_week]

        Returns:
            Combined temporal embeddings
            Shape: (batch_size, features, num_nodes, 1)
        """
        # Extract hour of day from the last time step
        day_emb = x[..., 1]
        day_index = (day_emb[:, -1, :] * self.time).clamp(0, self.time - 1).long()
        time_day = self.time_day[day_index]
        time_day = time_day.transpose(1, 2).unsqueeze(-1)

        # Extract day of week from the last time step
        week_emb = x[..., 2]
        week_index = week_emb[:, -1, :].clamp(0, 6).long()
        time_week = self.time_week[week_index]
        time_week = time_week.transpose(1, 2).unsqueeze(-1)

        # Combine temporal embeddings
        tem_emb = time_day + time_week

        return tem_emb
