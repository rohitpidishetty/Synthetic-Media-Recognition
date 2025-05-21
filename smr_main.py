import sys
import pickle
import metrics
import numpy as np
from metrics import evaluate
from face_extractor import dlibCNN
from keras.models import load_model
from VGG_16_Attention import nn_model
from dataset import celeb_df, faceforensic_plusplus

BATCH_SIZE = 20
EPOCHS = 30

if sys.argv[1] == '-celeb_df':
  celebdf = celeb_df()
  dlib_cnn = dlibCNN(IMG_HEIGHT=176, IMG_WIDTH=208, pre_trained_mmod="mmod_human_face_detector.dat")

  TRAINING_FEATURES_PATH = './DlibCNN_features/celeb_df_training_features.pkl'
  TRAINING_LABELS_PATH = './DlibCNN_features/celeb_df_training_labels.pkl'
  VALIDATION_FEATURES_PATH = './DlibCNN_features/celeb_df_validation_features.pkl'
  VALIDATION_LABELS_PATH = './DlibCNN_features/celeb_df_validation_labels.pkl'
  TESTING_FEATURES_PATH = './DlibCNN_features/celeb_df_testing_features.pkl'
  TESTING_LABELS_PATH = './DlibCNN_features/celeb_df_testing_labels.pkl'
  PREDICTIONS_PATH = './DlibCNN_features/celeb_df_predictions.pkl'

  if sys.argv[2] == '-dpp':
    celeb_train, celeb_test = celebdf.data_loader(path="../Datasets/Celeb-DF/", threshold=0.8)
    celeb_train_data, celeb_train_labels = celebdf.training_loader(celeb_train)
    celeb_train_features = []
    for td in celeb_train_data:
      celeb_train_features.append(dlib_cnn.srm_pre_processor(td))
    file = open(TRAINING_FEATURES_PATH, 'wb')
    pickle.dump(celeb_train_features, file)
    file.close()
    training_features = []
    training_labels = []
    for idx in range(len(celeb_train_features)):
      if celeb_train_features[idx] is not None:
        training_features.append(celeb_train_features[idx])
        training_labels.append(celeb_train_labels[idx])
    final_training_lables = []
    training_features = np.array(training_features)
    print(training_features.shape)
    for idx in range(len(training_features)):
      final_training_lables.extend([training_labels[idx] for x in range(len(training_features[idx]))])
    final_training_frames = np.concatenate(training_features, axis=0) if training_features else np.array([]) 
    final_training_lables = np.array(final_training_lables)
    validation_data = np.array(final_training_frames[:int(len(final_training_frames) * 0.2)])
    validation_labels = np.array(final_training_lables[:int(len(final_training_lables) * 0.2)])
    final_training_lables = np.array(final_training_lables)
    file = open(TRAINING_LABELS_PATH, 'wb')
    pickle.dump(final_training_lables, file)
    file.close()
    testing_features = []
    testing_labels = []
    celeb_test_data, celeb_test_labels = celebdf.test_loader(celeb_test)
    for idx in range(len(celeb_test_data)):
      if celeb_test_data[idx] is not None:
        testing_features.append(celeb_test_data[idx])
        testing_labels.append(celeb_test_labels[idx])
    final_testing_lables = []
    for idx in range(len(testing_features)):
      final_testing_lables.extend([testing_labels[idx] for x in range(len(testing_features[idx]))])
    final_testing_lables = np.array(final_testing_lables)
    final_testing_frames = np.concatenate(testing_features, axis=0) if testing_features else np.array([]) 
    file = open(TESTING_FEATURES_PATH, 'wb')
    pickle.dump(final_testing_frames, file)
    file.close()
    file = open(TESTING_LABELS_PATH, 'wb')
    pickle.dump(testing_labels, file)
    file.close()


  elif sys.argv[2] == '-lppd' and sys.argv[3] == '-train':
    file = open(TRAINING_FEATURES_PATH, 'rb')
    training_features = pickle.load(file)
    file.close()
    training_features = training_features
    file = open(TRAINING_LABELS_PATH, 'rb')
    training_labels = pickle.load(file)
    file.close()
    # Convert 3D matrices to 2D matices associated with labels
    # print(training_features.shape)
    # (8210, 176, 208, 3)
    # print(training_labels.shape)
    # (8210, )
    file = open(VALIDATION_FEATURES_PATH, 'rb')
    validation_features = pickle.load(file)
    file.close()
    file = open(VALIDATION_LABELS_PATH, 'rb')
    validation_labels = pickle.load(file)
    file.close()
    # print(validation_features.shape)
    # (1642, 176, 208, 3)
    # print(validation_labels.shape)
    # (1642, )
    vgg_model = nn_model()
    vgg_16_model = vgg_model.VGG_16_with_attention(input_shape=(176, 208, 3), num_classes=2)
    history = vgg_16_model.fit(
      training_features, training_labels, 
      batch_size=BATCH_SIZE, 
      epochs=EPOCHS, 
      validation_data=(validation_features, validation_labels), 
      callbacks=[vgg_model.celeb.model_checkpoint],
      verbose=1
    )

  elif sys.argv[2] == '-lm' and sys.argv[3] == '-test':
    model = load_model("./dl_models/smr_celeb_df_DLIB_CNN_SGD.keras")
    logs = open('./training_logs/celeb_df_training_logs.txt', 'r')
    # import re
    # for data in logs:
    #   if data.strip().lower().startswith('epoch'):
    #     if re.findall('Epoch[\\s]+[0-9]+/[0-9]+',data.strip()) == []:
    #       print(data.strip())
    #   if re.findall('(\d+)/(\d+)\s+━+\s+(\d+)s\s+(\d+)s/step - accuracy:\s([\d.]+)\s-\sloss:\s([\d.]+)\s-\sval_accuracy:\s([\d.]+)\s-\sval_loss:\s([\d.]+)', data.strip()):
    #     print(data.strip())
    file = open(TESTING_FEATURES_PATH, 'rb')
    test_frames = pickle.load(file)
    file.close()
    file = open(TESTING_LABELS_PATH, 'rb')
    test_labels = pickle.load(file)
    file.close()
    print(test_frames.shape)
    print(test_labels.shape)
    # predictions = model.predict(test_frames, batch_size=20, verbose=1)
    # print(predictions)
    # file = open(PREDICTIONS_PATH, 'wb')
    # pickle.dump(predictions, file)
    file = open(PREDICTIONS_PATH, 'rb')
    preds = np.array([pred.argmax() for pred in pickle.load(file)])
    file.close()
    eval_metrics = evaluate(test_labels, preds)
    print(eval_metrics)
    '''
    {
      'accuracy': 97.07744763760351, 
      'precision': 96.49532710280374, 
      'recall': 100.0, 
      'f1': 98.21640903686088, 
      'auc': 92.51870324189527, 
      'conf_matrix': 
        array(
          [
            [ 341,   60],
            [   0, 1652]
          ]
        )
    }
    '''

elif sys.argv[1] == '-ffpp':
  ffpp = faceforensic_plusplus()
  dlib_cnn = dlibCNN(IMG_HEIGHT=176, IMG_WIDTH=208, pre_trained_mmod="mmod_human_face_detector.dat")

  TRAINING_FEATURES_PATH = './DlibCNN_features/ffpp_training_features.pkl'
  TRAINING_LABELS_PATH = './DlibCNN_features/ffpp_training_labels.pkl'
  VALIDATION_FEATURES_PATH = './DlibCNN_features/ffpp_validation_features.pkl'
  VALIDATION_LABELS_PATH = './DlibCNN_features/ffpp_validation_labels.pkl'
  TESTING_FEATURES_PATH = './DlibCNN_features/ffpp_testing_features.pkl'
  TESTING_LABELS_PATH = './DlibCNN_features/ffpp_testing_labels.pkl'
  PREDICTIONS_PATH = './DlibCNN_features/ffpp_predictions.pkl'


  if sys.argv[2] == '-dpp':
    ffpp_train, ffpp_test = ffpp.data_loader(path="../Datasets/FF/FF_pp/", threshold=0.8)
    ffpp_train_data, ffpp_train_labels = ffpp.training_loader(ffpp_train)
    ffpp_train_features = []
    for td in ffpp_train_data:
      ffpp_train_features.append(dlib_cnn.srm_pre_processor(td))
    file = open(TRAINING_FEATURES_PATH, 'wb')
    pickle.dump(ffpp_train_features, file)
    file.close()
    training_features = []
    training_labels = []
    for idx in range(len(ffpp_train_features)):
      if ffpp_train_features[idx] is not None:
        training_features.append(ffpp_train_features[idx])
        training_labels.append(ffpp_train_labels[idx])
    final_training_lables = []
    training_features = np.array(training_features)
    print(training_features.shape)
    for idx in range(len(training_features)):
      final_training_lables.extend([training_labels[idx] for x in range(len(training_features[idx]))])
    final_training_frames = np.concatenate(training_features, axis=0) if training_features else np.array([]) 
    final_training_lables = np.array(final_training_lables)
    validation_data = np.array(final_training_frames[:int(len(final_training_frames) * 0.2)])
    validation_labels = np.array(final_training_lables[:int(len(final_training_lables) * 0.2)])
    final_training_lables = np.array(final_training_lables)
    file = open(TRAINING_LABELS_PATH, 'wb')
    pickle.dump(final_training_lables, file)
    file.close()
    testing_features = []
    testing_labels = []
    ffpp_test_data, ffpp_test_labels = ffpp.test_loader(ffpp_test)
    for idx in range(len(ffpp_test_data)):
      if ffpp_test_data[idx] is not None:
        testing_features.append(ffpp_test_data[idx])
        testing_labels.append(ffpp_test_labels[idx])
    final_testing_lables = []
    for idx in range(len(testing_features)):
      final_testing_lables.extend([testing_labels[idx] for x in range(len(testing_features[idx]))])
    final_testing_lables = np.array(final_testing_lables)
    final_testing_frames = np.concatenate(testing_features, axis=0) if testing_features else np.array([]) 
    file = open(TESTING_FEATURES_PATH, 'wb')
    pickle.dump(final_testing_frames, file)
    file.close()
    file = open(TESTING_LABELS_PATH, 'wb')
    pickle.dump(testing_labels, file)
    file.close()

  elif sys.argv[2] == '-lppd' and sys.argv[3] == '-train':
    file = open(TRAINING_FEATURES_PATH, 'rb')
    training_features = pickle.load(file)
    file.close()
    training_features = training_features
    file = open(TRAINING_LABELS_PATH, 'rb')
    training_labels = pickle.load(file)
    file.close()
    # Convert 3D matrices to 2D matices associated with labels
    # print(training_features.shape)
    # (1377, 176, 208, 3)
    # print(training_labels.shape)
    # (1377,)
    file = open(VALIDATION_FEATURES_PATH, 'rb')
    validation_features = pickle.load(file)
    file.close()
    file = open(VALIDATION_LABELS_PATH, 'rb')
    validation_labels = pickle.load(file)
    file.close()
    # print(validation_features.shape)
    # (115, 176, 208, 3)
    # print(validation_labels.shape)
    # (115,)
    vgg_model = nn_model()
    vgg_16_model = vgg_model.VGG_16_with_attention(input_shape=(176, 208, 3), num_classes=2)
    history = vgg_16_model.fit(
      training_features, training_labels, 
      batch_size=BATCH_SIZE, 
      epochs=EPOCHS, 
      validation_data=(validation_features, validation_labels), 
      callbacks=[vgg_model.ffpp.model_checkpoint],
      verbose=1
    )

  elif sys.argv[2] == '-lm' and sys.argv[3] == '-test':
    model = load_model("./dl_models/smr_ffpp_DLIB_CNN_SGD.keras")
    logs = open('./training_logs/ffpp_training_logs.txt', 'r')
    # import re
    # for data in logs:
    #   if data.strip().lower().startswith('epoch'):
    #     if re.findall('Epoch[\\s]+[0-9]+/[0-9]+',data.strip()) == []:
    #       print(data.strip())
    #   if re.findall('(\d+)/(\d+)\s+━+\s+(\d+)s\s+(\d+)s/step - accuracy:\s([\d.]+)\s-\sloss:\s([\d.]+)\s-\sval_accuracy:\s([\d.]+)\s-\sval_loss:\s([\d.]+)', data.strip()):
    #     print(data.strip())
    file = open(TESTING_FEATURES_PATH, 'rb')
    test_frames = pickle.load(file)
    file.close()
    file = open(TESTING_LABELS_PATH, 'rb')
    test_labels = pickle.load(file)
    file.close()
    print(test_frames.shape)
    # (153, 176, 208, 3)
    print(test_labels.shape)
    # (153,)
    # predictions = model.predict(test_frames, batch_size=20, verbose=1)
    # print(predictions)
    # file = open(PREDICTIONS_PATH, 'wb')
    # pickle.dump(predictions, file)
    file = open(PREDICTIONS_PATH, 'rb')
    preds = np.array([pred.argmax() for pred in pickle.load(file)])
    file.close()
    eval_metrics = evaluate(test_labels, preds)
    print(eval_metrics)
    '''
    {
      'accuracy': 92.81045751633987, 
      'precision': 92.81045751633987, 
      'recall': 100.0, 
      'f1': 96.27118644067797, 
      'auc': 50.0, 
      'conf_matrix': 
        array(
          [
            [  0,  11],
            [  0, 142]
          ]
        )
    }
    '''
