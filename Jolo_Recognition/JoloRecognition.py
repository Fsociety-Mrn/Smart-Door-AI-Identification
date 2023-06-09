import torch
import torchvision.transforms as transforms

from torchvision import datasets
from torch.utils.data import DataLoader
from facenet_pytorch import MTCNN,InceptionResnetV1



class JoloRecognition:
    
    def __init__(self):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # face detection
        self.mtcnn  = MTCNN(image_size=160, margin=0, min_face_size=40,select_largest=False, device=self.device)
        
        # facial recognition
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # load known faces data
        self.Saved_Data = torch.load('Model/data.pt', map_location='cpu')
        self.Embeding_List = self.Saved_Data[0]
        self.Name_List = self.Saved_Data[1]
    
    # for face recognition
    def Face_Compare(self, face, threshold=0.6):
    
        with torch.no_grad():
            
            # check if there is detected faces
            face,prob = self.mtcnn(face, return_prob=True)
            
            # check if there is face and probability of 90%
            if face  is not None and prob > 0.90:
                
                # calculcate the face distance
                emb  = self.facenet(face.unsqueeze(0)).detach()
                
                match_list = []
                
                # self.Embeding_List is the load data.pt 
                for idx, emb_db in enumerate(self.Embeding_List):

                    # torch.dist = is use to compare the face detected into batch of faceas in self embediing
                    dist = torch.dist(emb, emb_db).item()
                    # print(dist)
                      
                    # append the comparing result
                    match_list.append(dist)

                
                # check if there is recognize faces               
                if len(match_list) > 0:
                    
                    # match_list is the result of comparing faces
                    min_dist = min(match_list)

                    # since it has result we need to setup the accuracy level 
                    # threshold is the bias point number for accuracy
                    # in this if statment we set a threshold value of 0.6
                    # meaning all the result of comparing faces should atleast 0.6 value in order to recognize people
                    # print("threshold value: ",min_dist )
                    if min_dist < threshold:
                        
                        idx_min = match_list.index(min_dist)
                        
                        # print(min_dist)
                        return (self.Name_List[idx_min], min_dist)
                    else:

                        return ('No match detected', None)
                
                else:
                    return ('No match detected', None)
                
            else:
                return ('No match detected', None)
    
    def Face_Multiple_Compare(self,face):
        
        results = []
        
        boxes,probs = self.mtcnn.detect(face)
        
        for box in boxes:
            x1,y1,x2,y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            
            # to face crop
            face_crop = face[y1:y2, x1:x2]

            result = self.Face_Compare(face=face_crop)

            # print("Result Name: ",result[0])
            # print("Result box: ", box)
            results.append((result[0],box))

        return results
            
            
            
    
    # training from dataset
    def Face_Train(self, Dataset_Folder="Known_Faces", location="Model"):
        try:

            cplusplus = 1
        # define a function to collate data
            def collate_fn(x):
                return x[0]
            
        # locate the dataset of known faces
            dataset = datasets.ImageFolder(Dataset_Folder)

        # load the folder name in dataset
            label_names = {i: c for c, i in dataset.class_to_idx.items()}

        # load the dataset
            loader = DataLoader(dataset, batch_size=20, collate_fn=collate_fn, pin_memory=True)

        # create empty lists for storing embeddings and names
            name_list = []
            embedding_list = []

            for images, label in loader:
                print(str(cplusplus) + " Training...")
                with torch.no_grad():

                # for facial detection level 2 --- Using MTCNN model
                    face, prob = self.mtcnn(images, return_prob=True)

                # check if there is a detected face and has probability of 90%
                    if face is not None and prob > 0.90:
                    # calculate face distance
                        emb = self.facenet(face.unsqueeze(0))

                        embedding_list.append(emb.detach())
                        name_list.append(label_names[label])

                        cplusplus +=1

            data = [embedding_list, name_list]

        # save the calculated face distance into data.pt
            torch.save(data, location + '/data.pt')

            return "Successfully trained"

        except Exception as e:
            print(f"Error occurred while training the model: {str(e)}")
            return "Error occurred while training the model"

# JoloRecognition().Face_Train()