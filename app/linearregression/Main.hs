{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE OverloadedRecordDot #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE BangPatterns      #-}
{-# LANGUAGE DeriveGeneric     #-}
{-# LANGUAGE LambdaCase        #-} --言語拡張　ネットで調べる　この文を消すとエラーが出る
{-# LANGUAGE OverloadedStrings #-}

-- readFile :: FilePath -> IO ByteString
module Main where
import Control.Applicative
import Control.Monad (when)
import Torch
-- import Text.CSV(Text)
import GHC.Generics(Generic)
import System.IO
import System.Exit (exitFailure)
-- from bytestring
import Data.ByteString (ByteString, hGetSome, empty)
import qualified Data.ByteString.Lazy as BL
-- from cassava
import Data.Csv 
import Data.Text (Text)
import qualified Data.Vector as V

import ML.Exp.Chart (drawLearningCurve) 
import qualified System.Random.Shuffle

data Temprature = Temprature {
  date :: !String,
  daily_mean_temprature :: !Float }
  deriving (Generic,Show)

-- instance FromRecord Temprature
-- instance ToRecord Temprature
instance FromNamedRecord Temprature where
    parseNamedRecord r = Temprature <$> r .: "date" <*> r .: "daily_mean_temprature"


model :: Linear -> Tensor -> Tensor
model state input = squeezeAll $ linear state input

-- groundTruth :: Tensor -> Tensor
-- groundTruth t = squeezeAll $ matmul t weight + bias
--   where
--     weight = asTensor ([42.0, 64.0, 96.0] :: [Float])
--     bias = full' [1] (3.14 :: Float) -- aがweight bがbias(aとbを変数とおく)
--     -- full :: forall shape dtype device a. (TensorOptions shape dtype device, Scalar a) => a -> Tensor device dtype shape

-- validで計算する
-- transpose2D:転置行列にする
groundTruth :: Tensor -> Linear -> Tensor
groundTruth t l = matmul t (transpose2D(toDependent l.weight)) + (toDependent l.bias)
  
printParams :: Linear -> IO () --パラメータを表示
printParams trained = do
  putStrLn $ "Parameters:\n" ++ (show $ toDependent $ trained.weight)
  putStrLn $ "Bias:\n" ++ (show $ toDependent $ trained.bias)

-- リストの最初の8個のFloatをとってきてその後1~7日のリストと8日目を追加するリストを作成する
make_7days_temperature_pair_list :: [Float] -> [([Float],Float)] -> [([Float],Float)]
make_7days_temperature_pair_list temperature_list sevendays_list
  | length temperature_list < 8 = sevendays_list
  | otherwise = make_7days_temperature_pair_list (Prelude.tail temperature_list) (sevendays_list ++ [(Prelude.take 7 temperature_list, last (Prelude.take 8 temperature_list))])-- 　temprature_listの最初から7個をsevendats_listに追加する


-- Temprature型を受け取ったらdaily_mean_tempratureを返す      
return_daily_mean_temprature :: Temprature -> Float
return_daily_mean_temprature = daily_mean_temprature

-- floatのリストとfloatのペアを受け取ったらfloatのりすとをを返す
pair_to_list :: ([Float],Float) -> [Float]
pair_to_list (list_,_) = list_

-- Vector Tempratureを受け取ったらfloatのリストを返す
-- toList :: Vector a -> [a]
make_float_list :: (V.Vector Temprature) -> [Float]
make_float_list vector_temprature =
  let temprature_list = V.toList vector_temprature
  in map return_daily_mean_temprature temprature_list


main :: IO ()
main = do
  -- ファイルを読み込む
  train <- BL.readFile "data/train.csv"
  

  -- float型の気温のみのリスト
  let train_temprature_list = case decodeByName train  of
        Left err -> []
        Right (_, v) -> make_float_list v
  -- 7日間の気温のリストのリスト
  let train_sevendays_temprature_pair_list = make_7days_temperature_pair_list train_temprature_list []
  let train_sevendays_temprature_list = map pair_to_list train_sevendays_temprature_pair_list
  -- print train_sevendays_temprature_list
  -- valid <- readFile("data/valid.csv")
  -- readFromFile ("data/train.csv")
  eval <- BL.readFile("data/eval.csv")
  let eval_temprature_list = case decodeByName eval of
        Left err -> []
        Right (_, v) -> make_float_list v
      eval_sevendays_temprature_pair_list = make_7days_temperature_pair_list eval_temprature_list []
      (eval_float_list, eval_y_float) = unzip eval_sevendays_temprature_pair_list
      eval_tensor_list = asTensor eval_float_list
      eval_tensor_y = asTensor eval_y_float
  valid <- BL.readFile("data/valid.csv")
  let valid_temprature_list = case decodeByName valid of
        Left err -> []
        Right (_, v) -> make_float_list v
      valid_sevendays_temprature_pair_list = make_7days_temperature_pair_list valid_temprature_list []
      (valid_float_list, valid_y_float) = unzip valid_sevendays_temprature_pair_list
      valid_tensor_list = asTensor valid_float_list
      valid_tensor_y = asTensor valid_y_float
  -- let eval_loss_list -- ロスを表示させるのに使う
  init <- sample $ LinearSpec {in_features = numFeatures, out_features = 1} -- sample parametrised classという型クラス　linear specモデルの設定を定義しているデータ型 
  -- infeatures 入ってくる次元を指定　
  randGen <- defaultRNG
  printParams init
  let perEpoch = Prelude.length train_sevendays_temprature_pair_list `Prelude.div` batchSize -- 1epochの間に実行する回数
  (trained, losses_list, valid_losses_list,pair_list) <- foldLoop (init, [],[],train_sevendays_temprature_pair_list) numIters $ \(state, loss_list,valid_loss_list,pair_list) i -> do
    -- let (input, randGen') = randn' [batchSize, numFeatures] randGen --変数を減らす　input:バッチサイズ*特徴量の配列
    
    let start_number = (i `mod` perEpoch)* batchSize
        end_number = start_number + batchSize
        input_list =  Prelude.take (end_number - start_number) (drop start_number pair_list)--　バッチサイズ分もらってくる 
        (input_float, y_float) = unzip input_list
        input = asTensor input_float
        y = asTensor y_float
        y' = model state input -- なってほしい値と出た値(8日目の気温)
        loss = mseLoss y y' --mseの誤差
    -- 1epochごとに
    -- 1. リストの順番をシャッフルする
    -- 2. validを用いてロスを計算
    when (i `mod` perEpoch == 0) $ do    
      let numEpoch = i `Prelude.div` perEpoch
      putStrLn $ "epoch: " ++ show numEpoch ++ " | Loss: " ++ show loss --　１００回ごとに誤差を表示
    (newParam, _) <- runStep state optimizer loss 1e-6 -- 更新してくれる関数 
    let new_loss_list = if i `mod` perEpoch == 0 then loss : loss_list 
                        else loss_list --エポック数が増えるごとにロスを更新
    new_shuffle_list <- if i `mod` perEpoch == 0 then System.Random.Shuffle.shuffleM pair_list
                        else pure pair_list
    let new_valid_loss_list = if i `mod` perEpoch == 0 
                              then let valid_y' = groundTruth (asTensor (Prelude.take batchSize valid_float_list)) newParam
                                       valid_y = asTensor (Prelude.take batchSize valid_y_float)
                                   in (mseLoss valid_y valid_y') : valid_loss_list
                              else valid_loss_list
    pure (newParam,new_loss_list,new_valid_loss_list,new_shuffle_list)    
  printParams trained
  -- save [(toDependent trained.weight)] "/home/acf16407il/.local/lib/hasktorch/bordeaux-intern2024/app/linearregression/parameters/Adagradweight.text"
  -- save [(toDependent trained.bias)] "/home/acf16407il/.local/lib/hasktorch/bordeaux-intern2024/app/linearregression/parameters/Adagradbias.text"
  drawLearningCurve "/home/acf16407il/.local/lib/hasktorch/bordeaux-intern2024/app/linearregression/image/Adagradloss.png" "Learning Curve" [("train_loss",map asValue (reverse losses_list)),("valid_loss",map asValue (reverse valid_losses_list))] 
  pure ()
  where
    optimizer = GD --勾配降下法
    defaultRNG = mkGenerator (Device CPU 0) 31415
    batchSize = 128
    numIters = 2000
    numFeatures = 7 -- optimiser で暗点に陥らないように、やり方を考える今回は勾配

  
  -- saveParams trainedModel "app/temperature/models/temp-model.pt"
  -- drawLearningCurve "app/temperature/curves/graph-avg.png" "Learning Curve" [("train_loss",reverse losses)]


-- エポックデータ全部を１回ずつみた単位？
-- バッチ学習　全部のデータを見る　ミニバッチ　一部
-- この単位でロスを計測したい
-- トレーニングにtrain.csv
-- 評価 eval
-- train.csvだけだと過学習をしすぎちゃうのでvalid トレーニング中にロスは取るけど、そのモデルを使ってロスの更新はしない 過学習が起こったかどうかをvalidデータで確認するグラフを出す