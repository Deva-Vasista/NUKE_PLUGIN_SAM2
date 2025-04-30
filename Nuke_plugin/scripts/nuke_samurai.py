# Removed original processing imports
import nuke
import nukescripts
from api_client import NukeSamuraiClient

class NukeSamuraiPanel(nukescripts.PythonPanel):
    def __init__(self):
        nukescripts.PythonPanel.__init__(self, 'NukeSamurai', 'uk.co.thefoundry.NukeSamurai')
        
        # Create UI elements
        self.client = NukeSamuraiClient()
        
        # API URL
        self.api_url = nuke.String_Knob('api_url', 'API URL', 'http://localhost:8000')
        self.addKnob(self.api_url)
        
        # Frame range
        self.start_frame = nuke.Int_Knob('start_frame', 'Start Frame')
        self.end_frame = nuke.Int_Knob('end_frame', 'End Frame')
        self.addKnob(self.start_frame)
        self.addKnob(self.end_frame)
        
        # Process button
        self.process_btn = nuke.PyScript_Knob('process', 'Process')
        self.process_btn.setCommand('process_exr()')
        self.addKnob(self.process_btn)
        
        # Status
        self.status = nuke.Text_Knob('status', 'Status', '')
        self.addKnob(self.status)
        
    def knobChanged(self, knob):
        if knob is self.process_btn:
            self.process_exr()
            
    def process_exr(self):
        try:
            # Update client with new API URL
            self.client.api_url = self.api_url.value()
            
            # Check server health
            health = self.client.check_health()
            if not health:
                self.status.setValue('Server not responding')
                return
                
            # Get selected node
            node = nuke.selectedNode()
            if not node:
                nuke.message('Please select a node to process')
                return
                
            # Get frame range
            start_frame = self.start_frame.value()
            end_frame = self.end_frame.value()
            
            # Process each frame
            for frame in range(start_frame, end_frame + 1):
                nuke.frame(frame)
                file_path = node['file'].value()
                
                # Create mask node
                mask_node = self.client.create_mask_node(file_path)
                if mask_node:
                    mask_node.setInput(0, node)
                    self.status.setValue(f'Processed frame {frame}')
                else:
                    self.status.setValue(f'Failed to process frame {frame}')
                    break
                    
        except Exception as e:
            self.status.setValue(f'Error: {str(e)}')
            nuke.message(f'Error processing EXR: {str(e)}')

# Add to Nuke menu
def add_to_menu():
    menu = nuke.menu('Nodes')
    menu.addCommand('NukeSamurai', lambda: NukeSamuraiPanel().showModalDialog())

# Register the panel
def register_panel():
    nukescripts.registerPanel('uk.co.thefoundry.NukeSamurai', add_to_menu)

# Initialize the plugin
register_panel()
