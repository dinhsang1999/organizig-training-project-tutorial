import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.dataset import RetinaDataset
from src.model import RetinaModel
from fastprogress import progress_bar

IMAGE_SIZE = 224
DATA_DIR = '/Users/oddphoton/Projects/vietai/vietai_advance_w1b_retinal_disease_classificaton'
DATA_DIR_TEST_IMAGES = os.path.join(DATA_DIR, 'test')
DATA_DIR_TEST_RESULTS = os.path.join(DATA_DIR, 'sample_submission.csv')
DATA_DIR_TRAIN_LABEL = os.path.join(DATA_DIR, 'train_sample.csv')

TRAINED_MODEL_PATH="models/retina_epoch2_score0.2500.pth"    #place your saved checkpoint file here.

labels = pd.read_csv(DATA_DIR_TRAIN_LABEL)
LABELS = labels.columns[1:]
NUM_CLASSES = labels.shape[1] - 1 # minus first column

#Create mapping between label and index
LABEL_TO_INDEX={}
INDEX_TO_LABEL={}
for index in np.arange(NUM_CLASSES):
  INDEX_TO_LABEL[index]=LABELS[index]
  LABEL_TO_INDEX[LABELS[index]]=index

def test(model, test_loader, device):
    model.eval()

    out_pred = torch.FloatTensor().to(device)

    with torch.no_grad():
        for step, (images, _, _) in enumerate(progress_bar(test_loader)):
            images = images.to(device)

            # Feed forward 
            pred = model(images)

            out_pred = torch.cat((out_pred, pred), 0)
    
    # clear memory
    del images
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_pred.to("cpu").numpy()

if __name__ == '__main__':
    # Load model
    print('----- Loading model ... -----')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RetinaModel(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH)['model'])
    print('Done')

    # Load test dataset
    print('----- Loading test data ... -----')
    test_df = pd.read_csv(DATA_DIR_TEST_RESULTS)
    test_dataset = RetinaDataset(DATA_DIR_TEST_IMAGES, test_df, (IMAGE_SIZE, IMAGE_SIZE), True)
    test_dataloader = DataLoader(dataset=test_dataset, 
                                batch_size=1, 
                                shuffle=None, 
                                num_workers=0,
                                pin_memory=True)
    print('Done')

    # Test
    print('----- Testing ... -----')
    out_pred = test(model, test_dataloader, device)
    out_pred = (out_pred > 0.5)*1
    print("- Number of tested image: ", len(test_df.filename))
    print('Done')

    # Write results
    print('----- Writing results ... -----')
    predicts_metadata = {}
    predicts_metadata['filename'] = test_df.filename 
    for i in np.arange(NUM_CLASSES):
        predicts_metadata[INDEX_TO_LABEL[i]] = out_pred[:,i]
    predicts_df = pd.DataFrame(predicts_metadata)
    print(predicts_df[1:5])
    
    predicted=[]
    for idx, row in predicts_df.iterrows():
        disease_index=[str(idx) for idx, value in enumerate(row[1:]) if value==1]
        #Reverse => sort decreasing
        disease_index=disease_index[::-1]
        #Convert to string
        disease_index=" ".join(disease_index) 
        predicted.append(disease_index)

    submission_metadata={}
    submission_metadata['filename']=predicts_df.filename
    submission_metadata['predicted']=predicted

    submission_df = pd.DataFrame(submission_metadata)
    submission_df.to_csv("results/submission.csv", index=False)
    print(submission_df[1:5])
    print('Done')