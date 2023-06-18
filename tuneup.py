def test_inductive(self,model = None, nodes = None, val_edges = None, k=50):
        model.eval()  # Set the model to evaluation mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # add this line to set the device

        # Take 5 samples from val_edges as positive examples
        if model == None:
            model = self.model

        if nodes == None:
            nodes = self.V_new

        if val_edges == None:
            val_edges = self.data_processor.E_val


        with torch.no_grad():

            # Embedding new nodes ------- DIFFERENT PART
            z = model(z, self.data_processor.E_val)

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
