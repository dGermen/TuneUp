import torch
import model
from torch_geometric.utils import negative_sampling
import numpy as np
import random
import time
import data_processor
from utils import *

class train_controller():

    def __init__(self, 
                 preferred_device=None,
                 SEED = 42,
                 V_percentage = 0.95,
                 train_edge_percentage = 0.5,
                 ) -> None:

        if preferred_device == None:
            if torch.backends.mps.is_available():
                preferred_device=       'mps'
            elif torch.cuda.is_available():
                preferred_device= 'cuda'
            else:
                preferred_device=  'cpu'

        self.device = torch.device(preferred_device)

        self.SEED = SEED
        self.seed(SEED)

        self.V_percentage = V_percentage
        self.train_edge_percentage = train_edge_percentage

        self.data_processor = data_processor.data_processor(self)

        self.num_nodes = self.data_processor.num_nodes
        self.V = self.data_processor.V
        self.start_node_dict_val = self.data_processor.start_node_dict_val
        


    def seed(self, SEED):
        # Seed everything for deterministic runs
        random.seed(SEED)
        np.random.seed(SEED)

        # Seed torch
        torch.manual_seed(SEED)


        # If you're using CUDA:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        
    def train(self, 
              model = None,
              V = None, 
              data = None, 
              train_edges = None, 
              val_edges = None, 
              optimizer = None, 
              patience=10, 
              epochs = 1000, 
              test_active = True, 
              save_path = None):
        
        if model == None:
            self.model = model.GCN(128)
        
        if V == None:
            V = self.V
        
        if data == None:
            data = self.data_processor.data

        if train_edges == None:
            train_edges = self.data_processor.E_train

        if val_edges == None:
            val_edges = self.data_processor.E_val

        if optimizer == None:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

        # Define some initial best validation loss as infinity
        best_val_loss = float('inf')
        epochs_no_improve = 0

        train_losses = []
        val_losses = []

        # Training loop
        data, train_edges, val_edges = data.to(self.device), train_edges.to(self.device), val_edges.to(self.device)
        for epoch in range(epochs):  # 1000 epochs
            print("epoch ", epoch)

            model.train()
            optimizer.zero_grad()

            z_train = model(data, train_edges)  # embeddings for training edges
            pos_edge_index = train_edges  # positive examples
            neg_edge_index = negative_sampling(edge_index=pos_edge_index, num_nodes=z_train.size(0))  # negative examples

            #print("pos_edge_index.shape: ", pos_edge_index.shape)
            pos_logit = model.decode(z_train, pos_edge_index)
            neg_logit = model.decode(z_train, neg_edge_index)

            loss = model.bpr_loss(pos_logit, neg_logit)
            # append the loss to the train_losses
            train_losses.append(loss)

            loss.backward()
            optimizer.step()

            print("train loss: ", loss.item())



            # Validation:
            if (epoch +1) % 5 == 0:
                # validation function calls model.eval(), calculating both val loss & recall@50
                val_loss = self.validation(model, V, val_edges, z_train)
                # append the val loss to the val_losses
                val_losses.append(val_loss)
                print(f'Validation Loss: {val_loss}')
                if test_active:
                    res = self.test_transductive(model, V, val_edges,z_train, 50)
                    print("recall@50: ", res)

                """
                # Check if early stopping conditions are met
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve == patience:
                        print(f'Early stopping triggered after {epoch+1} epochs.')
                        break
                """


            # save the model @ each 200 epochs
            if (epoch +1) % 100 == 0:
                save_all(model, train_losses, val_losses, epoch +1, save_path)

        # save @ exit
        save_all(model, train_losses, val_losses, epochs, save_path)


    def validation(self,model, nodes, val_edges, z):
        #model.eval()  # Set the model to evaluation mode

        with torch.no_grad():


            pos_edge_index = val_edges  # positive examples
            neg_edge_index = negative_sampling(edge_index=pos_edge_index, num_nodes=z.size(0))  # negative examples


            # Negative examples for validation

            pos_logit = model.decode(z, pos_edge_index)
            neg_logit = model.decode(z, neg_edge_index)

            val_loss = model.bpr_loss(pos_logit, neg_logit)

        return val_loss.item()
    
    def test_transductive(self,model, nodes, val_edges, z, k=50):
        model.eval()  # Set the model to evaluation mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # add this line to set the device

        # Take 5 samples from val_edges as positive examples


        with torch.no_grad():

            # Convert V to a boolean tensor for faster lookup.
            v_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
            v_mask[nodes] = True
            v_mask = v_mask.to(device)

            # Assume val_edges contains the validation edges (it should be a 2 x num_val_edges tensor)
            # val_edges = ...

            # Check if both nodes of each edge in val_edges are in V
            source_nodes = val_edges[0, :]
            target_nodes = val_edges[1, :]
            can_exist_in_V = v_mask[source_nodes] & v_mask[target_nodes]

            can_exist_in_V.to(device)

            # Filter the edges that can exist in V
            valid_edges_in_V = val_edges[:, can_exist_in_V].to(device)
            positive_pairs = valid_edges_in_V


            # FOR MEMORY
            selected_pairs = positive_pairs[:, torch.randint(valid_edges_in_V.size(1), (500,))]


            # --- Generating negative pairs ---

            # Find the unique starting nodes in val_edges
            start_nodes = torch.unique(selected_pairs[0, :]).to(device)

            # Initialize a numpy array to keep node id and recall@50 for each node
            recall_at_k = np.zeros((len(start_nodes), 2))

            # Tour over start nodes
            a = time.time()
            for start_node in start_nodes:
                timezzz = time.time()


                all_possible_pairs = torch.stack(torch.meshgrid(start_node, self.V), dim=-1).reshape(-1, 2).t().to(device)

                # Clock time for look up and print it
                start_node = int(start_node)
                positive_pairs = self.start_node_dict_val[start_node] # THIS IS HARDCODED GLOBAL VARIABLE DO NOT COPY PASTE


                # Remove the existing edges in val_edges from all_possible_pairs to create the negative pairs
                existing_pairs = positive_pairs.t()
                existing_pairs = existing_pairs.to(device)

                # Removing positive pairs that are generated accidentaly
                negative_pairs = remove_common_edges(E_all=positive_pairs,B=all_possible_pairs) # B - (A INTERSECTION B)

                # Negative examples for validation
                timezzzz = time.time()
                positive_scores = model.decode(z, positive_pairs)
                negative_scores = model.decode(z, negative_pairs)


                # Combine positive edges and negative scores
                all_edges = torch.cat([positive_pairs, negative_pairs], dim=1)
                all_scores = torch.cat([positive_scores, negative_scores])
                # Indicate which edges are positive (1 for positive, 0 for negative)
                positive_edge_indicator = torch.tensor([1]*positive_pairs.size(1) + [0]*negative_pairs.size(1)).to(device)


                recall= calculate_recall_per_node(all_edges, all_scores, positive_edge_indicator, k,start_node)
                # Report passed time

                recall_at_k[start_node, 0] = start_node
                recall_at_k[start_node, 1] = recall

                del all_possible_pairs
                del all_scores
                del all_edges

            return recall, recall_at_k
