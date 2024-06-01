from HuggingMouse.pipelines.base import Pipeline
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from pathlib import Path
import os
from HuggingMouse.utils import make_container_dict
import pandas as pd
from HuggingMouse.make_embeddings import MakeEmbeddings
import pickle
import plotly.express as px
from transformers import AutoImageProcessor


class NeuronPredictionPipeline(Pipeline):
    def __init__(self, **kwargs):
        self.model = kwargs['model']
        self.model_name_str = self.model.name_or_path
        self.model_prefix = self.model.name_or_path.replace('/', '_')
        self.regression_model = kwargs['regression_model']
        self.single_trial_f = kwargs['single_trial_f']
        self.test_set_size = kwargs.get('test_set_size', 0.7)
        allen_cache_path = os.environ.get('HGMS_ALLEN_CACHE_PATH')
        if allen_cache_path is None:
            pass
            # raise AllenCachePathNotSpecifiedError()
        transformer_embedding_cache_path = os.environ.get(
            'HGMS_TRANSF_EMBEDDING_PATH')
        if transformer_embedding_cache_path is None:
            pass
            # raise TransformerEmbeddingCachePathNotSpecifiedError()
        self.regr_analysis_results_cache = os.environ.get(
            'HGMS_REGR_ANAL_PATH')
        if self.regr_analysis_results_cache is None:
            pass
            # raise RegressionOutputCachePathNotSpecifiedError()
        self.boc = BrainObservatoryCache(manifest_file=str(
            Path(allen_cache_path) / Path('brain_observatory_manifest.json')))
        self.eid_dict = make_container_dict(self.boc)
        self.stimulus_session_dict = {
            'three_session_A': ['natural_movie_one', 'natural_movie_three'],
            'three_session_B': ['natural_movie_one'],
            'three_session_C': ['natural_movie_one', 'natural_movie_two'],
            'three_session_C2': ['natural_movie_one', 'natural_movie_two']
        }
        embedding_file_path = os.path.join(
            transformer_embedding_cache_path, f"{self.model_prefix}_embeddings.pkl")
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_name_str)
        if not os.path.exists(embedding_file_path):
            self.embeddings = MakeEmbeddings(
                self.processor, self.model).execute()
        else:
            with open(embedding_file_path, 'rb') as f:
                self.embeddings = pickle.load(f)

    def __call__(self, container_id) -> str:
        self.current_container = container_id
        # Create an empty DataFrame to hold the merged data
        merged_data = None
        for session, _ in self.stimulus_session_dict.items():
            try:
                session_dct = self.handle_single_session(
                    container_id, session)
                session_df = pd.DataFrame(session_dct)
                if merged_data is None:
                    merged_data = session_df
                else:
                    # Perform an outer join on 'cell_id' column
                    merged_data = pd.merge(
                        merged_data, session_df, on='cell_ids', how='outer')
            except Exception as e:
                print(f'Error: {e}')
        self.merged_data = merged_data

        return self

    def handle_single_session(self, container_id, session):
        session_eid = self.eid_dict[container_id][session]
        dataset = self.boc.get_ophys_experiment_data(session_eid)
        cell_ids = dataset.get_cell_specimen_ids()
        dff_traces = dataset.get_dff_traces()[1]
        session_stimuli = self.stimulus_session_dict[session]
        session_dct = pd.DataFrame()
        session_dct['cell_ids'] = cell_ids
        # Compile the sessions into the same column to avoind NAN's
        # and make the data processing a bit easier
        if session == 'three_session_C2':
            sess = 'three_session_C'
        else:
            sess = session
        data_dct = {
            'dff_traces': dff_traces,
            'test_set_size': self.test_set_size,
            'regression_model': self.regression_model
        }
        for s in session_stimuli:
            data_dct['movie_stim_table'] = dataset.get_stimulus_table(s)
            data_dct['embedding'] = self.embeddings[s]
            # There are only 10 trials in each session-stimulus pair
            for trial in range(10):
                data_dct['trial'] = trial
                return_dict = self.single_trial_f(**data_dct)
                for key, value in return_dict.items():
                    if key == 'scores':
                        for sc in value:
                            session_dct[f'{sess}_{s}_{trial}_{sc}'] = value[sc]
        print(session_dct)
        return session_dct

    def heatmap(self, args=None):
        print('plotting')

        # Exclude the 'cell_ids' column from the heatmap
        merged_data_no_ids = self.merged_data.drop(columns=['cell_ids'])

        # Clip the values to have a minimum of -1 for the heatmap
        merged_data_clipped = merged_data_no_ids.clip(lower=-1)

        # Create the heatmap with clipped values
        fig = px.imshow(merged_data_clipped,
                        labels=dict(x="Trials", y="Neurons", color="Score"),
                        x=merged_data_clipped.columns,
                        y=merged_data_clipped.index,
                        color_continuous_scale='BuPu'
                        )

        # Update x-axis to place it on the top
        fig.update_xaxes(side="top")

        # Add custom hover text
        hover_text = [[f'cell_id: {cell_id}<br>Session, trial and metric: {trial}<br>Score: {score}'
                       for trial, score, cell_id in zip(self.merged_data.columns[1:], row[1:], [int(row['cell_ids'])] * len(self.merged_data.columns[1:]))]
                      for idx, row in self.merged_data.iterrows()]

        fig.data[0].update(
            hovertemplate='%{customdata}',
            customdata=hover_text
        )

        fig.show()
        return self

    def dropna(self, args=None):
        self.merged_data.dropna(inplace=True)
        return self

    def filter_data(self, args=None):
        print('filtering')
        return self

    def scatter_movies(self, args=None):
        print('plotting')

        neuron_ids = self.merged_data['cell_ids']

        movie_one_df = self.merged_data.filter(like='natural_movie_one')
        movie_two_df = self.merged_data.filter(like='natural_movie_two')
        movie_three_df = self.merged_data.filter(like='natural_movie_three')

        # Calculate the mean for each row
        row_means_one = movie_one_df.mean(axis=1)
        row_means_two = movie_two_df.mean(axis=1)
        row_means_three = movie_three_df.mean(axis=1)

        std_one = movie_one_df.std(axis=1)
        std_two = movie_two_df.std(axis=1)
        std_three = movie_three_df.std(axis=1)

        # Create a DataFrame for the means
        means_df = pd.DataFrame(
            {'Mean_one': row_means_one, 'Mean_two': row_means_two, 'Mean_three': row_means_three, 'cell_id': neuron_ids, 'Var exp for movie one': std_one, 'Var exp for movie two': std_two, 'Var exp for movie three': std_three})

        # Create a 3D scatter plot with Plotly
        fig = px.scatter_3d(means_df, x='Mean_one', y='Mean_two', z='Mean_three',
                            labels={'Mean_one': 'Mean var exp for movie one',
                                    'Mean_two': 'Mean var exp for movie two', 'Mean_three': 'Mean var exp for movie three'},
                            title='3D Scatter Plot of Mean Variance Explained Across Stimuli',
                            hover_data={'Var exp for movie one': True, 'Var exp for movie two': True,
                                        'Var exp for movie three': True, 'cell_id': True},
                            color_discrete_sequence=['aquamarine'])  # 'rgb(144, 238, 144)'])

        # Show the plot
        fig.show()
        return self
