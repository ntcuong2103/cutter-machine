import torch.nn as nn

class LSTM_With_ControlParam(nn.Module):
    def __init__(self, input_size=1, control_param_size=6, hidden_size=64, num_layers=1, output_size=3):
        super(LSTM_With_ControlParam, self).__init__()

        # self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # có thể thay bằng 1 lớp linear
        self.linear1 = nn.Linear(input_size, hidden_size * 2) # W (1 x hidden_size*2)
        
        # embedding: two linear layers with ReLU activation in between
        self.embedding = nn.Sequential(
            nn.Linear(control_param_size, hidden_size * 4),  # increase dimension for better representation
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size * 2)  # match LSTM output dimension
        )

        # có thể tăng num_layers = 2
        self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        # self.embedding = nn.Linear(control_param_size, hidden_size * 2)  # optional embedding for control parameters
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 for bidirectional

    def forward(self, x, control_param):
        # x: (B, T, 1), control_param: (B, 6)
        # lstm_out, _ = self.lstm1(x)  # lstm_out: (B, T, hidden_size * 2)
        lstm_out = self.linear1(x)  # (B, T, hidden_size * 2)
        control_param = self.embedding(control_param)  # (B, hidden_size * 2)
        combined = lstm_out + control_param.unsqueeze(1)  # broadcast control_param across time steps
        
        lstm_out2, _ = self.lstm2(combined)  # (B, T, hidden_size * 2)
        output = self.fc(lstm_out2)  # (B, output_size)
        return output

# lightning model definition
import torch
import pytorch_lightning as pl

class LitLSTM(pl.LightningModule):
    def __init__(self, input_size=1, control_param_size=6, hidden_size=64, num_layers=2, output_size=3, learning_rate=1e-3):
        super(LitLSTM, self).__init__()
        self.model = LSTM_With_ControlParam(input_size, control_param_size, hidden_size, num_layers, output_size)
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate

    def forward(self, x, control_param):
        return self.model(x, control_param)

    def training_step(self, batch, batch_idx):
        x = batch['x']  # (B, T, 1) (Batch, Time, Features)
        y = batch['y']  # (B, T, 3)
        v = batch['v']  # (B, 6)
        x_len = batch['x_len']  # (B,)
        y_pred = self(x, v)  # (B, T, 3)
        
        # compute loss only on the valid time steps (before padding)
        y_pred = [y_pred[i, :x_len[i]] for i in range(len(x_len))]
        y = [y[i, :x_len[i]] for i in range(len(x_len))]
        y_pred = torch.cat(y_pred, dim=0)
        y = torch.cat(y, dim=0)

        loss = self.criterion(y_pred, y)  # compare with the last time step of y

        # add loss for y0, y1, y2
        loss_y0 = self.criterion(y_pred[:, 0], y[:, 0])
        loss_y1 = self.criterion(y_pred[:, 1], y[:, 1])
        loss_y2 = self.criterion(y_pred[:, 2], y[:, 2])

        self.log('train_loss', loss)
        self.log('train_loss_y0', loss_y0)
        self.log('train_loss_y1', loss_y1)
        self.log('train_loss_y2', loss_y2)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['x']  # (B, T, 1) (Batch, Time, Features)
        y = batch['y']  # (B, T, 3)
        v = batch['v']  # (B, 6)
        x_len = batch['x_len']  # (B,)
        y_pred = self(x, v)  # (B, T, 3)
        
        # compute loss only on the valid time steps (before padding)
        y_pred = [y_pred[i, :x_len[i]] for i in range(len(x_len))]
        y = [y[i, :x_len[i]] for i in range(len(x_len))]
        y_pred = torch.cat(y_pred, dim=0)
        y = torch.cat(y, dim=0)

        loss = self.criterion(y_pred, y)  # compare with the last time step of y

        # add loss for y0, y1, y2
        loss_y0 = self.criterion(y_pred[:, 0], y[:, 0])
        loss_y1 = self.criterion(y_pred[:, 1], y[:, 1])
        loss_y2 = self.criterion(y_pred[:, 2], y[:, 2])

        self.log('val_loss', loss)
        self.log('val_loss_y0', loss_y0)
        self.log('val_loss_y1', loss_y1)
        self.log('val_loss_y2', loss_y2)

    def test_step(self, batch, batch_idx):
        x = batch['x']  # (B, T, 1) (Batch, Time, Features)
        y = batch['y']  # (B, T, 3)
        v = batch['v']  # (B, 6)
        x_len = batch['x_len']  # (B,)
        y_pred = self(x, v)  # (B, T, 3)

        # compute loss only on the valid time steps (before padding)
        y_pred = [y_pred[i, :x_len[i]] for i in range(len(x_len))]
        y = [y[i, :x_len[i]] for i in range(len(x_len))]
        y_pred = torch.cat(y_pred, dim=0)
        y = torch.cat(y, dim=0)

        loss = self.criterion(y_pred, y)  # compare with the last time step of y

        # add loss for y0, y1, y2
        loss_y0 = self.criterion(y_pred[:, 0], y[:, 0])
        loss_y1 = self.criterion(y_pred[:, 1], y[:, 1])
        loss_y2 = self.criterion(y_pred[:, 2], y[:, 2])

        self.log('test_loss', loss)
        self.log('test_loss_y0', loss_y0)
        self.log('test_loss_y1', loss_y1)
        self.log('test_loss_y2', loss_y2)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

# main as test
if __name__ == "__main__":
    model = LSTM_With_ControlParam()
    x = torch.randn(2, 500, 1)  # (B, T, 1)
    v = torch.randn(2, 6)        # (B, 6)
    y_pred = model(x, v)         # (B, T, 3)
    print(y_pred.shape)         # should be (2, 500, 3)