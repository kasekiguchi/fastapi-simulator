
def calc_input(self,xh):
    # フィードバック
    self.u = float(self.F @ (-xh))
    return self.u
