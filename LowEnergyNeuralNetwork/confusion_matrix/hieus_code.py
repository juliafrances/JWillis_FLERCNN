from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix

def plot_confusion(true_y, pred_y, experiment_dir, plot_name, labels=None):
  '''
  Plot and save confusion matrices.
  '''
  conf_matrix_pred = confusion_matrix(true_y, pred_y, normalize='pred').transpose()
  conf_matrix_true = confusion_matrix(true_y, pred_y, normalize='true').transpose()
  # Plot
  plt.clf()
  cmap = 'Blues'
  #colors = ["#FFCCFF", "#F1DAFF", "#E3E8FF", "#CCFFFF"]
  #cmap = matplotlib.colors.ListedColormap(colors)
  fig, axes = plt.subplots(figsize=(12,7), nrows=1, ncols=2)
  im = axes[0].imshow(conf_matrix_pred, interpolation='nearest', cmap=cmap)
  axes[0].set_title("Normalized on prediction")
  for (j,i),label in np.ndenumerate(conf_matrix_pred):
    axes[0].text(i,j,"{:>.3}".format(label),ha='center',va='center')
  im = axes[1].imshow(conf_matrix_true, interpolation='nearest', cmap=cmap)
  axes[1].set_title("Normalized on truth")
  for (j,i),label in np.ndenumerate(conf_matrix_true):
    axes[1].text(i,j,"{:>.3}".format(label),ha='center',va='center')
  # Style
  ticks = np.arange(conf_matrix_pred.shape[0])
  for ax in axes.flat:
    ax.set_ylabel("Predicted label")
    ax.set_xlabel("True label")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, rotation=90, verticalalignment='center')
  fig.suptitle("Confusion matrices\n"+plot_name, size='x-large', linespacing = 1.5)
  fig.subplots_adjust(right=0.8, top=1.02)
  cbar_ax = fig.add_axes([0.84, 0.308, 0.02, 0.528])
  fig.colorbar(im, cax=cbar_ax)
  #Save
  plotfile = os.path.join(experiment_dir, 'conf_matrix.png')
  plt.savefig(plotfile)
  plt.close(fig)