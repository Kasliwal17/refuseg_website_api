from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from model import Unet
from processing import pre_process_brats, post_process_brats
import gdown

def download_weights():
    url = 'https://drive.google.com/file/d/1Bm57B2jzs4RikRoWzGXlLtvIt_5bJuG3/view?usp=sharing'
    output = 'refuseg_beta_1.pth'
    gdown.download(url, output, quiet=False)

download_weights()

model =Unet(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        encoder_depth = 4,
        classes=4,
        activation=None,
        in_channels=1,
        contrastive=False
    )
device = torch.device('cpu')
state_dict = torch.load('./refuseg_beta_1.pth',map_location='cpu')
state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()

# Code from: https://fastapi.tiangolo.com/tutorial/request-files/
app = FastAPI()
origins = [
    "http://www.refuseg.tech",
    "https://www.refuseg.tech",
    "http://76.76.21.21",
    "https://76.76.21.21",
    "https://refuseg-website.vercel.app/"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/uploadfiles/")
async def create_upload_files(t1: UploadFile = File(None), t1c: UploadFile = File(None),
                              t2: UploadFile = File(None), flair: UploadFile = File(None)):
    """ Create API endpoint to send images to and specify
    what type of file it'll take

    :return: A mask in png format
    :rtype: StreamingResponse
    """

    # Assuming a function like "pre_process_brats" exists to handle all 4 files
    t1,tc,t2,flair = pre_process_brats(t1, t1c, t2, flair)

    # Run the model and post-process the output
    with torch.no_grad():
        prediction = model(t1,tc,t2,flair)

    # Assuming a function "post_process_brats" to post-process the mask
    output_image = post_process_brats(prediction)

    return StreamingResponse(output_image, media_type="image/png")


@app.get("/")
async def main():
    content = """
    <body>
          <h3>Upload images to get their processed version</h3>
          <form action="/uploadfiles/" enctype="multipart/form-data" method="post">
              <input name="files" type="file" multiple>
              <input type="submit">
          </form>
      </body>
    """
    return HTMLResponse(content=content)
