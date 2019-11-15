from torch import nn

def parse_cfg():
    """ Read in author's yolov3 config file. """
    with open('./yolov3.cfg', 'r') as cfg:
        lines = cfg.readlines()
        # Clean out empty lines and comments.
        lines = [x.strip() for x in lines if len(x.strip()) > 0 and x[0] != '#']

        block = {}
        blocks = []

        for line in lines:
            print(line)
            if line[0] == '[':
                if block != {}:
                    blocks.append(block)
                    block = {}
                block['type'] = line[1:-1].strip()
            else:  # This
                key, val = line.split('=')
                block[key.strip()] = val.strip()
        blocks.append(block)

        return blocks
# Layers required for YOLO
class DummyLayer(nn.Module):  # Placeholder for route/shortcut
    def __init__(self):
        super(DummyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def yolov3(blocks):
    """ Create a YOLOv3 network based on the config provided. """
    layers = nn.ModuleList()
    net = blocks[0]
    prev_channels = 3
    out_filters = []
    for i, block in enumerate(blocks):
        layer = []
        if block['type'] == 'convolutional':
            out_channels = int(block['filters'])
            layer.append(nn.Conv2d(
                in_channels=prev_channels,
                out_channels=out_channels,
                kernel_size=int(block['size']),
                stride=int(block['stride']),
                padding=(int(block['size'])-1 // 2) if int(block['pad']) else 0,
                bias=('batch_normalize' not in block.keys())
            ))
            if 'batch_normalize' in block.keys():
                layer.append(nn.BatchNorm2d(out_channels))
            if block['activation'] == 'leaky':
                layer.append(nn.LeakyReLU(0.1, inplace=True))
        elif block['type'] == 'upsample':
            layer.append(nn.Upsample(2, mode='bilinear'))
        elif block['type'] == 'route':
            block['layers'] = [int(x) for x in block['layers'].split(',')]
            r_start = block['layers'][0]
            if len(block['layers']) > 1:
                r_end = block['layers'][1]
            else:
                r_end = 0

            if r_start > 0: r_start = r_start - i
            if r_end > 0:   r_end = r_end - i
            route = DummyLayer()
            layer.append(route)

            if r_end < 0:
                out_channels = out_filters[i + r_start] + out_filters[i + r_end]
            else:
                out_channels = out_filters[i + r_start]
        elif block['type'] == 'shortcut':
            layer.append(DummyLayer())
        elif block['type'] == 'yolo':
            mask = block['mask'].split(',')
            mask = [int(x) for x in mask]
            anchors = block['anchors'].split(',')
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            layer.append(DetectionLayer([anchors[i] for i in mask]))

        layers.append(nn.Sequential(*layer))
        prev_channels = out_channels
        out_filters.append(out_channels)

    return (net, layers)

if __name__ == '__main__':
    model = yolov3(parse_cfg())
    print(model)
