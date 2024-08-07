{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE OverloadedRecordDot #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE BangPatterns      #-}
{-# LANGUAGE DeriveGeneric     #-}
{-# LANGUAGE LambdaCase        #-} --言語拡張　ネットで調べる　この文を消すとエラーが出る
{-# LANGUAGE OverloadedStrings #-}

module Main where
import Control.Applicative
import Control.Monad (when)
import Data.List (foldl', intersperse, scanl')
import Torch
import Data.Maybe
import GHC.Generics
import System.IO
import System.Exit (exitFailure)
import Data.ByteString (ByteString, hGetSome, empty)
import qualified Data.ByteString.Lazy as BL
-- from cassava
import Data.Csv 
import Data.Text (Text)
import qualified Data.Vector as V

import ML.Exp.Chart (drawLearningCurve) 
import qualified System.Random.Shuffle
--------------------------------------------------------------------------------
-- MLP
--------------------------------------------------------------------------------


data Titanic = Titanic
  { passengerId :: Maybe Float
  , survived :: Maybe Float
  , pclass :: Maybe Float
  , name :: Maybe String
  , sex :: Maybe String
  , age :: Maybe Float
  , sibSp :: Maybe Float
  , parch :: Maybe Float
  , ticket :: Maybe String
  , fare :: Maybe Float
  , cabin :: Maybe String
  , embarked :: Maybe String
  } deriving (Generic, Show)


data NewTitanic = NewTitanic
  { survived :: Float
  , pclass :: Float
  , sex :: Float
  , age :: Float
  , sibSp :: Float
  , parch :: Float
  , fare :: Float
  , embarked :: Float
  } deriving (Generic, Show)

-- 
data MLPSpec = MLPSpec
  { feature_counts :: [Int],
    nonlinearitySpec :: Tensor -> Tensor
  }

data MLP = MLP -- linearが重みとバイアス
  { layers :: [Linear], -- 層ごとに重みとバイアスの値を入れる
    nonlinearity :: Tensor -> Tensor
  }
  deriving (Generic, Parameterized)

-- string型をfloat型にする
stringSexToFloat :: String -> Float
stringSexToFloat s =
  if s == "male" then 1.0
  else 0.0

stringEmbarkedToFloat :: String -> Float
stringEmbarkedToFloat e =
  if e=="Q" then 0.0
  else if e=="S" then 1.0
  else 2.0

-- 空データがないかどうか確認する
isFull :: Titanic -> Bool
isFull Titanic{..} =
  isJust(survived) &&
  isJust(pclass) &&
  isJust(sex) &&
  isJust(age) &&
  isJust(sibSp) &&
  isJust(parch) &&
  isJust(fare) &&
  isJust(embarked)



titanicToNewTitanic :: Titanic -> NewTitanic
titanicToNewTitanic t =
  NewTitanic {
    survived = head (maybeToList t.survived),
    pclass = head (maybeToList t.pclass),
    sex = stringSexToFloat (head (maybeToList t.sex)),
    age = head (maybeToList t.age),
    sibSp = head (maybeToList t.sibSp),
    parch = head (maybeToList t.parch),
    fare = head (maybeToList t.fare),
    embarked = stringEmbarkedToFloat (head (maybeToList t.embarked))
  } --かけている値を平均値で埋める

deleteEmptyDataRow :: [Titanic] -> [NewTitanic]
deleteEmptyDataRow [] = []
deleteEmptyDataRow (t:titanic) =
  if isFull t then titanicToNewTitanic t : deleteEmptyDataRow titanic
  else deleteEmptyDataRow titanic
  

  
makeNewTitanicList :: (V.Vector Titanic) -> [NewTitanic]
makeNewTitanicList titanic_vector =
  let titanic_list = V.toList titanic_vector
  in deleteEmptyDataRow titanic_list

newTitanicToPair :: NewTitanic -> (Float,[Float])
newTitanicToPair nt =
  (nt.survived,[nt.pclass,nt.sex,nt.age,nt.sibSp,nt.parch,nt.fare,nt.embarked])

newTitanicToPairList :: [NewTitanic] -> [(Float,[Float])]
newTitanicToPairList newtitaniclist = map newTitanicToPair newtitaniclist


instance FromNamedRecord Titanic where
    parseNamedRecord r = 
      Titanic <$> r .: "PassengerId" <*> r .: "Survived" <*> r .: "Pclass" <*> r .: "Name" <*> r .: "Sex" <*> r .: "Age" <*> r .: "SibSp" <*> r .: "Parch" <*> r .: "Ticket" <*> r .: "Fare" <*> r .: "Cabin" <*> r .: "Embarked"
-- toNamedRecordのインスタンスにする　passengerIdとSurvivedとペアを[a] decodeByName
instance Randomizable MLPSpec MLP where 
  sample MLPSpec {..} = do --sample-MLPSpecからMLPを作る関数
    let layer_sizes = mkLayerSizes feature_counts -- [(a,b),(b,c)]
    linears <- mapM sample $ map (uncurry LinearSpec) layer_sizes --それぞれにリニアスペック
    return $ MLP {layers = linears, nonlinearity = nonlinearitySpec}
    where 
      mkLayerSizes (a : (b : t)) = 
        scanl shift (a, b) t
        where
          shift (a, b) c = (b, c)

mlp :: MLP -> Tensor -> Tensor --ひっくり返す intersperse nonlinearity（非線形関数の名前）を間に入れる関数　
mlp MLP {..} input = foldl' revApply input $ intersperse nonlinearity $ map linear layers
  where
    revApply x f = f x
--  map linear layers layerひとつずつlinear型にする

-- tpは真の陽性
-- fpは偽の陽性
-- fnは偽の陰性
calculateprecision :: Float -> Float -> Float
calculateprecision tp fp = 
  tp / (tp+fp)

calculaterecall :: Float -> Float -> Float
calculaterecall tp fn =
  tp/(tp+fn)

calculateaccuracy :: Float -> Float -> Float -> Float -> Float
calculateaccuracy tp fp tn fn =
  (tp+tn)/(tp+fp+tn+fn)

calculateF1Score :: Float -> Float -> Float
calculateF1Score prec reca =
  (2*prec*reca)/(prec+reca)

calculatemacroF1Score :: Float -> Float -> Float
calculatemacroF1Score fir sec =
  (fir + sec)/2

calculatemicroF1Score :: Float -> Float -> Float -> Float -> Float
calculatemicroF1Score tp fp tn fn =
  (tp+tn)/(tp+fp+tn+fn)

calculateweightedf1score :: Float -> Float -> Float -> Float -> Float -> Float -> Float
calculateweightedf1score fir sec tp fp tn fn =
  fir*((tp+fp)/(tp+fp+tn+fn))+sec*fir*((tn+fn)/(tp+fp+tn+fn))


count :: [Float] -> [Float] -> Float -> Float -> Float
count [] [] b c = 0.0
count (ac:acl) (ca:cal) a c =
  let cabi = if ca >= 0.5 then 1.0
             else 0.0
  in
  if ac == a && cabi == c
    then (count acl cal a c) + 1.0
  else count acl cal a c
--------------------------------------------------------------------------------
--         predic_sur predic_dea
-- act_sur     tp         fn
-- act_dea     fp         tn
---------------------------------------------------------------------------------


makeConfusionMatrix :: [Float] -> [Float] -> [[Float]]
makeConfusionMatrix aclist calist =
  let tp = count aclist calist 1 1 
      fp = count aclist calist 0 1
      tn = count aclist calist 0 0
      fn = count aclist calist 1 0
  in [[tp,fp],[tn,fn]]
--------------------------------------------------------------------------------
-- Training code
---------------------------------------------------------------------------------

batchSize = 128

-- predictを計算してくれる
model :: MLP -> Tensor -> Tensor
model params t = mlp params t

main :: IO ()
main = do
  train <- BL.readFile "app/xor-mlp/data/titanic/train.csv"
  let train_titanic_list = case decodeByName train  of
        Left err -> []
        Right (_, v) -> makeNewTitanicList v

  let list = newTitanicToPairList train_titanic_list
      test_list = Prelude.take batchSize list -- 8:2 9:1
      input_list = Prelude.drop batchSize list
      
      perEpoch = ((Prelude.length input_list) `Prelude.div` batchSize) -1
  init <-
    sample $
      MLPSpec
        { feature_counts = [7, 128, 1], --　入力層中間層出力層ごとの特徴量の数
          nonlinearitySpec = Torch.tanh -- Torch.sigmoid --　活性化関数
        } -- input 2 hidden 2 output 1

  -- loadParams model filePath = do
  -- tensors <- load filePath -- filepathからtensorのリストを読み込む
  -- let params = map IndependentTensor tensors-- tensor一つ一つIndependentTensor型にする
  -- pure $ replaceParameters model params -- paramsにモデルを置き換える
      
  trainedmodel <- loadParams init "/home/acf16407il/.local/lib/hasktorch/bordeaux-intern2024/app/xor-mlp/parameters/b"
  let (ty_float,test_float) = unzip test_list
      test = asTensor test_float
  --   -- print(test)
  let (ty, ty') = (asTensor ty_float, squeezeAll $ model trainedmodel test)
      testloss = mseLoss ty ty' 
  let float_ty' =  asValue ty' :: [Float]
  let aclist = ty_float 
      calist = float_ty'
  let tps = count aclist calist 1 1 
      fps = count aclist calist 0 1
      tns = count aclist calist 0 0
      fns = count aclist calist 1 0
  let tnd = count aclist calist 1 1 
      fnd = count aclist calist 0 1
      tpd = count aclist calist 0 0
      fpd = count aclist calist 1 0

  let f1s = calculateF1Score (calculateprecision tps fps) (calculaterecall tps fns)
      f1d = calculateF1Score (calculateprecision tpd fpd) (calculaterecall tpd fnd)

  
  putStrLn("survived recall:" ++ show (calculaterecall tps fns) ++ " precision:" ++ show (calculateprecision tps fps) ++ " f1score:" ++ show (f1s))
  putStrLn("dead recall:" ++ show (calculaterecall tpd fnd) ++ " precision:" ++ show (calculateprecision tpd fpd) ++ " f1score:" ++ show (f1d))
  print("confusion matrix")
  print(show (makeConfusionMatrix ty_float float_ty'))
  print("accuracy")
  print(show (calculateaccuracy tps fps tns fns)) 
  putStrLn("macro-F1Score:" ++ show (calculatemacroF1Score f1s f1d) ++ "micro-F1Score:" ++ show (calculatemicroF1Score tps fps tns fns) ++ "weightned-F1Score:" ++ show (calculateweightedf1score f1s f1d tps fps tns fns))
  

  return ()
  -- where
    -- optimizer = GD
    -- epoch = 200 --あとで1000に
  
    -- tensorXOR t = (1 - (1 - a) * (1 - b)) * (1 - (a * b)) -- 1-a:not a a*b:and　と考えると (not(not a and not b)and(not(a and b))
    --   where
    --     a = select 1 0 t --  0番目をとってくる
    --     b = select 1 1 t --  1番目をとってくる
    -- 一番精度の良さそうなモデルをピックアップする