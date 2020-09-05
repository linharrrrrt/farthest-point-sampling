import numpy as np

class FPS:
    def __init__(self, pcd_xyz, n_samples):
        self.n_samples = n_samples
        self.pcd_xyz = pcd_xyz
        self.n_pts = pcd_xyz.shape[0]
        self.dim = pcd_xyz.shape[1]
        self.selected_pts = None
        self.selected_pts_expanded = np.zeros(shape=(n_samples, 1, self.dim))
        self.remaining_pts = np.copy(pcd_xyz)
        self.grouping_radius = None
        self.dist_pts_to_selected = None  # Iteratively updated in step(). Finally re-used in group()
        self.labels = None
        # Random pick a start
        self.start_idx = np.random.randint(low=0, high=self.n_pts - 1)
        self.selected_pts_expanded[0] = self.remaining_pts[self.start_idx]
        self.n_selected_pts = 1
        self.dist_pts_to_selected_min = None
        self.res_selected_idx=None

    def get_selected_pts(self):
        self.selected_pts = np.squeeze(self.selected_pts_expanded, axis=1)
        return self.selected_pts

    def step(self):
        print(self.n_selected_pts)
        if self.n_selected_pts == 1:
            self.dist_pts_to_selected = self.__distance__(self.remaining_pts, self.selected_pts_expanded[:self.n_selected_pts]).T
            self.dist_pts_to_selected_min = np.min(self.dist_pts_to_selected, axis=1, keepdims=True)
            self.res_selected_idx = np.argmax(self.dist_pts_to_selected_min)
            self.selected_pts_expanded[self.n_selected_pts] = self.remaining_pts[self.res_selected_idx]
            self.n_selected_pts += 1


        elif self.n_selected_pts < self.n_samples:
            self.dist_pts_to_selected = self.__distance__(self.remaining_pts, np.expand_dims(np.expand_dims(self.remaining_pts[self.res_selected_idx],0),0)).T 
            for i in range(0,self.remaining_pts.shape[0]):
                if self.dist_pts_to_selected_min[i]>self.dist_pts_to_selected[i]:
                    self.dist_pts_to_selected_min[i]=self.dist_pts_to_selected[i]
            self.res_selected_idx = np.argmax(self.dist_pts_to_selected_min)
            self.selected_pts_expanded[self.n_selected_pts] = self.remaining_pts[self.res_selected_idx]
            self.n_selected_pts += 1
        else:
            print("Got enough number samples")


    def fit(self):
        for _ in range(1, self.n_samples):
            self.step()
            print("sampleing no.",_," point")
        return self.get_selected_pts()

    def group(self, radius):
        self.grouping_radius = radius   # the grouping radius is not actually used
        dists = self.dist_pts_to_selected

        # Ignore the "points"-"selected" relations if it's larger than the radius
        dists = np.where(dists > radius, dists+1000000*radius, dists)

        # Find the relation with the smallest distance.
        # NOTE: the smallest distance may still larger than the radius.
        self.labels = np.argmin(dists, axis=1)
        return self.labels


    @staticmethod
    def __distance__(a, b):
        return np.linalg.norm(a - b, ord=2, axis=2)
