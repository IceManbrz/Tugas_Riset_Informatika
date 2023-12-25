# Load the trained model
saver.restore(session, 'C:/Users/Path/handgest_1')

# Function to calculate the classification accuracy on the test set
def print_test_accuracy():
    num_test = len(data.valid.labels)
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    i = 0
    while i < num_test:
        j = min(i + batch_size, num_test)

        images = data.valid.images[i:j, :]
        labels = data.valid.labels[i:j, :]

        feed_dict = {x: images, y_true: labels}
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        i = j

    cls_true = np.argmax(data.valid.labels, axis=1)
    correct = (cls_true == cls_pred)
    correct_sum = correct.sum()
    acc = float(correct_sum) / num_test

    print("Accuracy on Validation-Set: {0:.1%} ({1} / {2})".format(acc, correct_sum, num_test))

# Print the accuracy on the test set
print_test_accuracy()
