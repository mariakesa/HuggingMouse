import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.base import clone


class VisualizerDimReduction:
    '''
    Class for visualizing trial averaged data in 3D with 
    a dimensionality reduction model provided to the constructor.
    The dimensionality reduction model can be any model that 
    adheres to the sklearn decomposition or manifold API (fit_transform
    method). 
    '''

    def __init__(self, dim_reduction_model):
        self.dim_reduction_model = dim_reduction_model
        self.stimulus_session_dict = {
            'three_session_A': ['natural_movie_one', 'natural_movie_three'],
            'three_session_B': ['natural_movie_one'],
            'three_session_C': ['natural_movie_one', 'natural_movie_two'],
            'three_session_C2': ['natural_movie_one', 'natural_movie_two']
        }

    def info(self):
        print('These are all the possible session stimulus pairs: ',
              self.stimulus_session_dict)

    def visualize(self, trial_averaged_data, session, stimulus):
        model = clone(self.dim_reduction_model)
        try:
            X_new = model.fit_transform(
                trial_averaged_data[str(session)][str(stimulus)])
            # Plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_new[:, 0], X_new[:, 1], X_new[:, 2],
                       c=range(X_new.shape[0]), marker='o', cmap='bwr')
            ax.set_xlabel('Neural Dim1')
            ax.set_ylabel('Neural Dim2')
            ax.set_zlabel('Neural Dim3')
            plt.title(
                f'Trial averaged dimensionality reduced plot for {session}-{stimulus}', wrap=True)
            plt.show()
        except:
            print(
                'This session stimulus combination doesn\'t exist! in this experiment container')
