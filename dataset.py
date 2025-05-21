from data_partitioner import fissure
import os

class celeb_df:
  def data_loader(self, path, threshold):
    partitioner = fissure()
    celeb_df_dataset_path = path
    celeb_df_dataset_contents = partitioner.ls(celeb_df_dataset_path)
    celeb_real_path = os.path.join(celeb_df_dataset_path, celeb_df_dataset_contents[0])
    celeb_fake_path = os.path.join(celeb_df_dataset_path, celeb_df_dataset_contents[2])
    celeb_real_content = partitioner.ls(celeb_real_path)
    celeb_fake_content = partitioner.ls(celeb_fake_path)
    self.celeb_real_path = celeb_real_path
    self.celeb_fake_path = celeb_fake_path
    return partitioner.train_test_split(celeb_real_content, celeb_fake_content, max_threshold=threshold)
  
  def training_loader(self, train):
    train_labels = train[0][1]
    train_labels.extend(train[1][1])
    train_data = [os.path.join(self.celeb_real_path, real_video) for real_video in train[0][0]]
    train_data.extend([os.path.join(self.celeb_fake_path, fake_video) for fake_video in train[1][0]])
    return train_data, train_labels
  
  def test_loader(self, test):
    test_labels = test[0][1]
    test_labels.extend(test[1][1])
    test_data = [os.path.join(self.celeb_real_path, real_video) for real_video in test[0][0]]
    test_data.extend([os.path.join(self.celeb_fake_path, fake_video) for fake_video in test[1][0]])
    return test_data, test_labels
  

class faceforensic_plusplus:
  def data_loader(self, path, threshold):
    partitioner = fissure()
    ffpp_dataset_path = path
    ffpp_dataset_contents = partitioner.ls(ffpp_dataset_path)
    ffpp_real_path = os.path.join(ffpp_dataset_path, ffpp_dataset_contents[0])
    ffpp_fake_path = os.path.join(ffpp_dataset_path, ffpp_dataset_contents[1])
    ffpp_real_content = partitioner.ls(ffpp_real_path)
    ffpp_fake_content = partitioner.ls(ffpp_fake_path)
    self.ffpp_real_path = ffpp_real_path
    self.ffpp_fake_path = ffpp_fake_path
    return partitioner.train_test_split(ffpp_real_content, ffpp_fake_content, max_threshold=threshold)
  
  def training_loader(self, train):
    train_labels = train[0][1]
    train_labels.extend(train[1][1])
    train_data = [os.path.join(self.ffpp_real_path, real_video) for real_video in train[0][0]]
    train_data.extend([os.path.join(self.ffpp_fake_path, fake_video) for fake_video in train[1][0]])
    return train_data, train_labels
  
  def test_loader(self, test):
    test_labels = test[0][1]
    test_labels.extend(test[1][1])
    test_data = [os.path.join(self.ffpp_real_path, real_video) for real_video in test[0][0]]
    test_data.extend([os.path.join(self.ffpp_fake_path, fake_video) for fake_video in test[1][0]])
    return test_data, test_labels