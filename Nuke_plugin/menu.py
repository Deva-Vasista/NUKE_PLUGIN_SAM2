import nuke

### SAM2
from scripts.NukeSAM2 import CreateSAM2Node, UpdatePath, GenerateMask, BoundingBox, InputInfos

m = nuke.menu('Nodes')
m = m.addCommand('SAMURAI', 'CreateSAM2Node()', tooltip="SAMURAI", icon="samurai_icon.png")
###