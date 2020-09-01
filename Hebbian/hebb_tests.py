out_size = (8, 12)
out_size_list = [out_size] if not hasattr(out_size, '__len__') else out_size
thingout_size = torch.tensor(out_size_list[0:min(len(out_size_list), 3)])
out_channels = thingout_size.prod().item()

map_radius = (self.out_size - 1) // 2