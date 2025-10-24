import torch.nn as nn
from typing import Optional
from graphxai.explainers._base import _BaseExplainer
from graphxai.utils import Explanation
import torch
from torch.nn import functional as F
import numpy as np
from myerson import MyersonSamplingExplainer, MyersonExplainer
from myerson import MyersonSamplingClassExplainer, MyersonClassExplainer
from myerson import ShapleySamplingClassExplainer, ShapleySamplingExplainer
from myerson import PerturbationClassExplainer, PerturbationExplainer
from torch_geometric.data import Data

# device = "cuda" if torch.cuda.is_available() else "cpu"

class MyersonExplainer_(_BaseExplainer):
    
    def __init__(self,
            model: nn.Module,
            emb_layer_name: Optional[str] = None,
            is_subgraphx: Optional[bool] = False,
            return_type='label',
        ):
        super().__init__(model, emb_layer_name, is_subgraphx)
        self.return_type=return_type
        self.explain_graph = True # Myerson values can only explain graph level explanations
        self.wrap_model()
    
    def wrap_model(self):
        self.model = WrappedModel(self.model, explain_graph=self.explain_graph,
                                  return_type=self.return_type)
        
    def _predict(self, x: torch.Tensor, edge_index: torch.Tensor,
                 forward_kwargs: dict = {}):
        return self.model(x, edge_index, **forward_kwargs)

    def get_explanation_graph(self, edge_index: torch.Tensor,
                              x: torch.Tensor, label: torch.Tensor, regression=False,
                              forward_kwargs: dict = {}):
        """
        Explain a whole-graph prediction.

        Args:
            edge_index (torch.Tensor, [2 x m]): edge index of the graph
            x (torch.Tensor, [n x d]): node features
            label (torch.Tensor, [n x ...]): labels to explain
            forward_kwargs (dict, optional): additional arguments to model.forward
                beyond x and edge_index
            which_class (int | None): for which class to get explanations, None
                for regression tasks

        Returns:
            exp (dict):
                exp['feature_imp'] (torch.Tensor, [d]): feature mask explanation
                exp['edge_imp'] (torch.Tensor, [m]): k-hop edge importance
                exp['node_imp'] (torch.Tensor, [m]): k-hop node importance
        """

        graph = Data(x, edge_index, y=label) 
        # if regression:
        explainer = MyersonSamplingExplainer(graph, self.model, disable_tqdm=True)
        my_values = explainer.sample_all_myerson_values()
        my_values = torch.tensor(list(my_values.values()))
        exp = Explanation(
            node_imp = my_values
        )
        exp.set_whole_graph(Data(x=x, edge_index=edge_index))
        return exp
        # else:
        #     explainer = MyersonSamplingClassExplainer(graph, self.model, disable_tqdm=True)
        #     my_values = explainer.sample_all_myerson_values()
        #     my_values = torch.tensor(np.array(list(my_values.values())))
        #     exps = [Explanation(node_imp = x) for x in my_values.T]
        #     for exp in exps: exp.set_whole_graph(Data(x=x, edge_index=edge_index))
        #     return exps

class ShapleyExplainer_(_BaseExplainer):
    
    def __init__(self,
            model: nn.Module,
            emb_layer_name: Optional[str] = None,
            is_subgraphx: Optional[bool] = False):
        super().__init__(model, emb_layer_name, is_subgraphx)
        self.explain_graph = True # Myerson values can only explain graph level explanations

    def get_explanation_graph(self, edge_index: torch.Tensor,
                              x: torch.Tensor, label: torch.Tensor, regression=False,
                              forward_kwargs: dict = {}):

        graph = Data(x, edge_index, y=label) 
        if regression:
            explainer = ShapleySamplingExplainer(graph, self.model, disable_tqdm=True)
            sh_values = explainer.sample_all_shapley_values()
            sh_values = torch.tensor(sh_values.values())
            exp = Explanation(
                node_imp = sh_values
            )
            exp.set_whole_graph(Data(x=x, edge_index=edge_index))
            return exp
        else:
            explainer = ShapleySamplingClassExplainer(graph, self.model, disable_tqdm=True)
            sh_values = explainer.sample_all_shapley_values()
            sh_values = torch.tensor(np.array(list(sh_values.values())))
            exps = [Explanation(node_imp = x) for x in sh_values.T]
            for exp in exps: exp.set_whole_graph(Data(x=x, edge_index=edge_index))
            return exps

class PerturbationExplainer_(_BaseExplainer):
    
    def __init__(self,
            model: nn.Module,
            emb_layer_name: Optional[str] = None,
            is_subgraphx: Optional[bool] = False):
        super().__init__(model, emb_layer_name, is_subgraphx)
        self.explain_graph = True # Myerson values can only explain graph level explanations

    def get_explanation_graph(self, edge_index: torch.Tensor,
                              x: torch.Tensor, label: torch.Tensor, regression=False,
                              forward_kwargs: dict = {}):

        graph = Data(x, edge_index, y=label) 
        if regression:
            explainer = PerturbationExplainer(graph, self.model, disable_tqdm=True)
            sh_values = explainer.calculate_all_perturbation_values()
            sh_values = torch.tensor(sh_values.values())
            exp = Explanation(
                node_imp = sh_values
            )
            exp.set_whole_graph(Data(x=x, edge_index=edge_index))
            return exp
        else:
            explainer = PerturbationClassExplainer(graph, self.model, disable_tqdm=True)
            sh_values = explainer.calculate_all_perturbation_values()
            sh_values = torch.tensor(np.array(list(sh_values.values())))
            exps = [Explanation(node_imp = x.squeeze()) for x in sh_values.T]
            for exp in exps: exp.set_whole_graph(Data(x=x, edge_index=edge_index))
            return exps

class WrappedModel(nn.Module):
    def __init__(self, model, explain_graph, return_type='label'):
        super().__init__()
        self.model = model
        self.explain_graph = explain_graph
        self.return_type = return_type
    def forward(self, x, edge_index, batch):
        with torch.no_grad():
            out = self.model(x, edge_index, batch)
            if self.return_type == 'label':
                out = out.argmax(dim=-1)
            elif self.return_type == 'prob':
                out = F.softmax(out, dim=-1)
            elif self.return_type == 'log_prob':
                out = F.log_softmax(out, dim=-1)
            else:
                raise ValueError("return_type must be 'label', 'prob', or 'log_prob'")

            if self.explain_graph:
                out = out.squeeze()

            return out