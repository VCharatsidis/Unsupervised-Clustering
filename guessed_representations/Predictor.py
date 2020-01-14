import torch


class Predictor(torch.nn.Module):
    def __init__(self, C_in, H_in, W_in):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Predictor, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.nonlinear = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(H, D_out)

    def neighbours(self, convolutions, perimeter):

        size = convolutions.shape[0]
        inputs = []

        for i in range(size):
            for j in range(size):
                total_input = 2 * perimeter + 1
                conv_loss_tensor = torch.zeros(total_input, total_input)

                for row in range(i - perimeter, i + perimeter + 1):
                    for col in range(j - perimeter, j + perimeter + 1):
                        if row >= 0 and row < size:
                            if col >= 0 and col < size:
                                conv_loss_tensor[row - (i - perimeter), col - (j - perimeter)] = convolutions[row, col]

                flatten_input = torch.flatten(conv_loss_tensor)
                inputs.append(flatten_input)

        return inputs


    def fast_neighbours(self, convolutions, perimeter):
        # TODO do something about the boarder.
        size = convolutions.shape[0]
        inputs = []

        row_dim = 2
        col_dim = 3
        for i in range(1, size-1):
            row_indeces = torch.tensor([i-1, i, i+1])
            for j in range(1, size-1):
                col_indeces = torch.tensor([j-1, j, j+1])
                col_elements = torch.index_select(convolutions, col_dim, col_indeces)
                neighbours = torch.index_selec(col_elements, row_dim, row_indeces)

        return inputs


    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h = self.linear1(x)
        nonlinear = self.nonlinear(h)

        y_pred = self.linear2(nonlinear)

        return y_pred
