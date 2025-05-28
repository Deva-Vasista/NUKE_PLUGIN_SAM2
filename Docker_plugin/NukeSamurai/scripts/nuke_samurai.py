import cv2
import nuke
import os
import threading
import gc 
import torch
from NukeSamurai.scripts.demo import main

nuke.tprint('Current thread : ' +str(threading.current_thread().name) )


class BoundingBox :
    input_path = None
    x = None
    y = None
    w = None
    h = None
    bounding_box_coord = None


    @classmethod
    def getBbox(cls):
        file_path = InputInfos.path
        input_file_name = str(os.path.splitext(os.path.basename(file_path))[0])
        
        if "%04d" in input_file_name :
          file_path = file_path.replace('%04d', str(f"{int(nuke.thisNode().knob('FrameRangeMin').value()):04}")) 
        elif "%03d" in input_file_name :
          file_path =  file_path.replace('%03d', str(f"{int(nuke.thisNode().knob('FrameRangeMin').value()):03}"))    

        cls.input_path = file_path
        img_path = cls.input_path

        img = cv2.imread(img_path,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        cv2.namedWindow("BBOX - Press Enter or Space to validate",0) 
        cv2.setWindowProperty("BBOX - Press Enter or Space to validate", cv2.WND_PROP_TOPMOST, 1)
        cv2.resizeWindow("BBOX - Press Enter or Space to validate", 1280, 720) 
        
        # Bounding Box window
        x,y,w,h = cv2.selectROI("BBOX - Press Enter or Space to validate",img, fromCenter=False, showCrosshair=False)

        cv2.waitKey(13)
        cv2.destroyWindow("BBOX - Press Enter or Space to validate") 

        cls.x = x
        cls.y = y
        cls.w = w
        cls.h = h
        cls.bounding_box_coord = x,y,w,h
        nuke.tprint(f"{x = } {y = }  {w = } {h = }")

        return x,y,w,h,cls.bounding_box_coord


class InputInfos :
    read = None
    path = None
    original_fps = None
    bits = None
    
    @classmethod
    def getInputInfos(cls):
        f = nuke.thisNode().dependencies()

        for i in f:
            cls.read = i

        # Get file path
        try :
            cls.path = cls.read.knob('file').getValue()
        except : 
            cls.path = ''

        # Get metadatas
        ## get input fps
        try :
            cls.original_fps = int(cls.read.metadata()['input/frame_rate'])  
        except :
            cls.original_fps = nuke.Root()['fps'].value()  

        ## Get input bit depth
        # Look for bitsperchannel into the metadatas 
        try : 
            cls.bits = cls.read.metadata()['input/bitsperchannel']   
        except :
            # Interpreting bit depth depending on the format
            nuke.tprint("No Bit Depth information in the input metadatas. Interpreting it from the input extension.")

            if cls.path.endswith("jpg") or cls.path.endswith("jpeg") or cls.path.endswith("png") or cls.path.endswith("tiff"):
                cls.bits = "8-bit fixed"
            elif cls.path.endswith("exr") :
                cls.bits = "32-bit float"
            else :
                TypeError("Cannot interpret bit depth from unsupported input format.")
       

def UpdatePath():
    InputInfos.getInputInfos()
    FilePath = InputInfos.path
    nuke.thisNode().knob('FilePath').setValue(FilePath)
    

def GenerateMask():
    Output_path = nuke.thisNode().knob('OutputPath').getValue()

    # Checks           
    if str(os.path.splitext(os.path.basename(Output_path))[0]) == '' :
        raise TypeError("You must assign a file name")
        
    if nuke.thisNode()['FilePath'].value().lower().endswith("mp4") :
        raise TypeError('Unsupported input format. Input must be an Image Sequence')
        
    if nuke.thisNode().knob('FileType').value() == "exr" :
        if ("%04d" not in Output_path)  :        
            if ("%03d"not in Output_path)  :       
                raise TypeError("Your file must contains '####' or '###'")
        

    video_path= nuke.thisNode().knob('FilePath').value()
    video_output_path=nuke.thisNode().knob('OutputPath').getValue()
    save_to_file=nuke.thisNode().knob('FileType').value()
    bbox_coord=BoundingBox.bounding_box_coord
    frame_range = [int(nuke.thisNode().knob('FrameRangeMin').value()) , int((nuke.thisNode().knob('FrameRangeMax').value()+ 1))]
    original_fps = int(InputInfos.original_fps)                        
    target_fps=int(nuke.thisNode().knob('FPS').value())
    bits = InputInfos.bits

    if nuke.thisNode().knob('ModelType').value() == 'Large' :
        model_path = "sam2_repo/checkpoints/sam2.1_hiera_large.pt"
    if nuke.thisNode().knob('ModelType').value() == 'Base+' :    
        model_path = "sam2_repo/checkpoints/sam2.1_hiera_base_plus.pt"
    if nuke.thisNode().knob('ModelType').value() == 'Small' :    
        model_path = "sam2_repo/checkpoints/sam2.1_hiera_small.pt"
    if nuke.thisNode().knob('ModelType').value() == 'Tiny' :    
        model_path = "sam2_repo/checkpoints/sam2.1_hiera_tiny.pt"

    # Define childThread
    childThread = threading.Thread(target=main, args=( 
        video_path,
        video_output_path,
        bbox_coord, 
        save_to_file,
        frame_range,
        original_fps,                        
        target_fps,
        bits,
        model_path
    ))
    # Starting thread
    childThread.start()
    
    gc.collect()
    torch.cuda.empty_cache()


def CreateSamuraiNode():

    # Creating node
    nuke.createNode('NoOp')
    s = nuke.selectedNode()

    # Adding knobs
    s.knob('name').setValue('SAMURAI')
    s.addKnob(nuke.File_Knob('FilePath', 'File Path'))
    s.addKnob(nuke.PyScript_Knob('UpdatePath', 'Update Path', 'UpdatePath()' ))
    s.addKnob(nuke.PyScript_Knob('CreateBoundingBox', 'Create Bounding Box', 'BoundingBox.getBbox()'))
   
    s.addKnob(nuke.Text_Knob(''))

    s.addKnob(nuke.Int_Knob("FPS", 'Output Frame Rate'))
    s.addKnob(nuke.Int_Knob("FrameRangeMin", 'Frame Range'))
    s.addKnob(nuke.Int_Knob("FrameRangeMax", ' '))
    s.addKnob(nuke.Enumeration_Knob('ModelType', 'Model type', ['Base+','Large', 'Small', 'Tiny']))

    s.addKnob(nuke.Text_Knob(' ', ''))
    
    s.addKnob(nuke.Enumeration_Knob('FileType', 'File type', ['exr', 'mp4']))
    s.addKnob(nuke.File_Knob('OutputPath', 'Output Path'))
    s.addKnob(nuke.PyScript_Knob('GenerateMask', 'Generate Mask', 'GenerateMask()'))
    
    
    
    
### SETTING RANGES, DEFAULT VALUES, TOOLTIPS & FORMAT ###
    s['FPS'].setValue(int(nuke.root().knob('fps').getValue()))
    s['FrameRangeMin'].setValue(int(nuke.Root()['first_frame'].value())) 
    s['FrameRangeMax'].setValue(int(nuke.Root()['last_frame'].value())) 


    s['FPS'].setFlag(nuke.STARTLINE)
    s['FrameRangeMax'].clearFlag(nuke.STARTLINE)
    s['UpdatePath'].setFlag(nuke.STARTLINE)
    s['GenerateMask'].setFlag(nuke.STARTLINE)
    
    s['CreateBoundingBox'].setTooltip("Create a bounding box. Press Enter or space to validate. Press C to cancel.")
    s['FPS'].setTooltip("Target FPS for the output video")
    s['ModelType'].setTooltip("Choose your model type")
    s['OutputPath'].setTooltip("path/to/your/file_####.exr, to create an image sequence add #### or ### ")
    s['GenerateMask'].setTooltip("Generate Mask")
    