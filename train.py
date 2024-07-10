import torch
import torch.nn as nn
from torch_geometric.graphgym import optim

from SynMapper.SynValid.aam_validator import AMMValidator
from SynMapper.model.datamodule import SynMapperDataModule
from SynMapper.model.model import GCNGenerator

file_name = 'benchmark'
data_module = SynMapperDataModule(file_name)

input_dim = 3
hidden_dims = [32, 64, 128, 256]
output_dim = input_dim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = GCNGenerator(input_dim, hidden_dims, output_dim).to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=0.001)

criterion = nn.BCELoss()

num_epochs = 200
real_labels = torch.ones((data_module.num_examples, 1)).to(device)

for epoch in range(num_epochs):
    generator.train()

    g_loss_epoch = 0
    for batch in data_module.train_loader(batch_size=32):
        batch = batch.to(device)

        optimizer_G.zero_grad()

        z = torch.randn((batch.num_nodes, input_dim)).to(device)
        fake_data = generator(z, batch.edge_index)

        reactions = fake_data.x
        ground_truth = batch.y

        classified_labels = AMMValidator.smiles_check(reactions, ground_truth)
        classified_labels = torch.tensor(classified_labels, dtype=torch.float).to(device).view(-1, 1)

        real_labels = torch.ones_like(classified_labels)

        g_loss = criterion(classified_labels, real_labels[:classified_labels.size(0)])

        g_loss.backward()
        optimizer_G.step()

        g_loss_epoch += g_loss.item()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, G Loss: {g_loss_epoch / len(data_module.train_split)}')

generator.eval()
with torch.no_grad():
    z = torch.randn((data_module.num_examples, input_dim)).to(device)
    for batch in data_module.test_loader(batch_size=32):
        batch = batch.to(device)
        generated_data = generator(z, batch.edge_index).cpu()
