import os

class fissure:
  
  def ls(self, path):
   return os.listdir(path)

  def train_test_split(self, real, fake, max_threshold):
    real_count = len(real)
    fake_count = len(fake)
    real_thold = int(real_count * max_threshold)
    real_train = real[:real_thold]
    real_test = real[real_thold:]
    fake_thold = int(fake_count * max_threshold)
    fake_train = fake[:fake_thold]
    fake_test = fake[fake_thold:]
    return ((real_train, [0 for real in range(len(real_train))]), (fake_train, [1 for fake in range(len(fake_train))])), ((real_test, [0 for real in range(len(real_test))]), (fake_test, [1 for fake in range(len(fake_test))]))