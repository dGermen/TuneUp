##########################
##### TuneUp w/o syn-tails
##########################







def tuneup(mode, model,  V, data, train_edges, val_edges, optimizer, epochs, test_active, drop_percent, save_path, save_name, epochs_finetune):

  if mode == "tuneup":
    # tune_up_ours(model,  V, data, train_edges, val_edges, optimizer, num_epochs_finetune, test_active_finetune, save_path, save_name, drop_percent=0.2)
    tune_up_ours(model,  V, data, train_edges, val_edges, optimizer, epochs, test_active, save_path, save_name, drop_percent=0.2)

  elif mode == "wo_synt":
    # tune_up_wo_syn_tails(V, data, train_edges, val_edges, optimizer, num_epochs_train, num_epochs_finetune, test_active, save_path, save_name)
    tune_up_wo_syn_tails(V, data, train_edges, val_edges, optimizer, epochs, epochs_finetune, test_active, save_path, save_name)

  elif mode == "wo_curr":
    # tune_up_wo_curriculum(V, data, train_edges, val_edges, optimizer, save_path, save_name, drop_percent=0.2, epochs = 1000, test_active = True)
    tune_up_wo_curriculum(V, data, train_edges, val_edges, optimizer, save_path, save_name, drop_percent=drop_percent, epochs = epochs, test_active = test_active)



def save_all(model, train_losses, val_losses, epoch, save_path, save_name):
  
  # save the model
  model_save_path = save_path + "/" +save_name + "_model_"+str(epoch)+".pt"
  print("model_save_path: ", model_save_path)
  torch.save(model.state_dict(), model_save_path)

  # save the train loss
  trainloss_save_path = save_path + "/" +save_name + "_train_losses_" + str(epoch) + ".pkl"
  with open(trainloss_save_path, 'wb') as f:
      pickle.dump(train_losses, f)

  # save the val loss
  valloss_save_path = save_path + "/" +save_name + "_val_losses_" + str(epoch) + ".pkl"
  with open(valloss_save_path, 'wb') as f:
      pickle.dump(val_losses, f)    