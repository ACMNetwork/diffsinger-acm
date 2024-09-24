import torch
import torch.nn as nn

class convPlugin(nn.Module):
	def __init__(self,dim):
		super().__init__()
		self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
		self.act = nn.GELU()
	def forward(self,x: torch.Tensor) -> torch.Tensor:
		return self.act(self.dwconv(x))

class MultiHeadConv(nn.Module):
	def __init__(self,dim):
		super().__init__()
		self.act7 = nn.GELU()
		self.conv2d = nn.Conv1d(dim * 3, dim, kernel_size=7, padding=3, groups=dim)
		self.stacking1 = nn.ModuleList(
			convPlugin(dim) for _ in range(3)
		)
		self.stacking2 = nn.ModuleList(
			convPlugin(dim) for _ in range(4)
		)
		self.stacking3 = nn.ModuleList(
			convPlugin(dim) for _ in range(4)
		)
	def forward(self,x: torch.Tensor) -> torch.Tensor:
		x1 = x
		x2 = x
		x3 = x
		for conv in self.stacking1:
			x1 = conv(x1)
		for conv in self.stacking2:
			x2 = conv(x2)
		for conv in self.stacking3:
			x2 = conv(x2)
		selected_tensor = torch.cat((x1,x2,x3),dim=1)
		selected_tensor = self.conv2d(selected_tensor)
		return self.act7(selected_tensor)
