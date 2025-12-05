furuta_pendulumでは以下の方針でファイル分けしています。 

/furuta_pendulum
|- FURUTA_PENDULUM : 制御対象のclassフォルダ 
|- CONTROLLER : 制御入力生成用クラスフォルダ 
|- simulator.py : simulator本体 各フォルダの構成は以下 

FURUTA_PENDULUM 
|- base.py : 本体クラス、State class, Parameter class 
|- get_param.py : 制御対象の物理パラメータを返すpublic method ：モデルエラーも表現可能 
|- apply_input.py : 入力、dtを受け取り1step 数値シミュレーションをした結果を返すpublic メソッド 
|- measure.py : センサーを模擬したpublic method y = h(x)のイメージ 
|- ode.py : 煩雑になりやすい微分方程式を独立させただけのmethod (private) 

CONTROLLER 
|- base.py : 本体クラス、State class, Parameter class 
|- Ac.py : parameter を受け線形近似モデルを返す 
|- Bc.py : parameter を受け線形近似モデルを返す 
|- calc_input.py : 具体的な入力を算出するmethod 
|- estimator.py : 状態推定するためのmethod (複雑なシステムの場合クラスとして独立する)