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

data Tempature = Tempature {
  date :: !String,
  daily_mean_temprature :: !Float }
  deriving (Generic,Show)

-- instance FromRecord Tempature
-- instance ToRecord Tempature
instance FromNamedRecord Tempature where
    parseNamedRecord r = Tempature <$> r .: "date" <*> r .: "daily_mean_temprature"
model :: Linear -> Tensor -> Tensor
model state input = squeezeAll $ linear state input

groundTruth :: Tensor -> Tensor
groundTruth t = squeezeAll $ matmul t weight + bias
  where
    weight = asTensor ([42.0, 64.0, 96.0] :: [Float])
    bias = full' [1] (3.14 :: Float)

printParams :: Linear -> IO ()
printParams trained = do
  putStrLn $ "Parameters:\n" ++ (show $ toDependent $ trained.weight)
  putStrLn $ "Bias:\n" ++ (show $ toDependent $ trained.bias)

-- readfile
-- feed :: (ByteString -> Parser Tempature) -> Handle -> IO (Parser Tempature)
-- feed k csvFile = do
--   hIsEOF csvFile >>= \case
--     True  -> return $ k empty
--     False -> k <$> hGetSome csvFile 4096


-- ファイルを読み込んで出力
-- readFromFile :: FilePath -> IO String
-- readFromFile filepath = do
--   withFile filepath ReadMode $ \ csvFile -> do
--     let loop !_ (Fail _ errMsg) = do putStrLn errMsg; exitFailure
--         loop acc (Many rs k)    = loop (acc <> rs) =<< feed k csvFile
--         loop acc (Done rs)      = print (acc <> rs)
        
--     loop [] (decode NoHeader)

--csv_to_array :: ByteString -> 

-- リストの最初の7個のFloatをとってきてそれを追加するリストを作成する
make_7days_tempature_list :: [Float] -> [Float] -> [Float]
make_7days_tempature_list [] sevendays_list = []
make_7days_tempature_list [x1] sevendays_list = []
make_7days_tempature_list [x1,x2] sevendays_list = []
make_7days_tempature_list [x1,x2,x3] sevendays_list = []
make_7days_tempature_list [x1,x2,x3,x4] sevendays_list = []
make_7days_tempature_list [x1,x2,x3,x4,x5] sevendays_list = []
make_7days_tempature_list [x1,x2,x3,x4,x5,x6] sevendays_list = []
make_7days_tempature_list [x1,x2,x3,x4,x5,x6,x7] sevendays_list = sevendays_list + [x1,x2,x3,x4,x5,x6,x7] --気温のリストの長さが７の時に終わり
make_7days_tempature_list tempature_list sevendays_list = make_7days_tempature_list (tail tempature_list) (sevendays_list + (Prelude.take 7 tempature_list)) 

-- Tempature型を受け取ったらdaily_mean_tempratureを返す                                                                               tempature_listの最初を抜いたリスト　tempature_listの最初から7個をsevendats_listに追加する
return_daily_mean_temprature :: Tempature -> Float
return_daily_mean_temprature = daily_mean_temprature

-- Vector Tempatureを受け取ったらfloatのリストを返す
-- toList :: Vector a -> [a]
make_float_list :: (V.Vector Tempature) -> [Float]
make_float_list vector_tempature =
  let tempature_list = V.toList vector_tempature
  in map return_daily_mean_temprature tempature_list


main :: IO ()
main = do
  -- ファイルを読み込む
  train <- BL.readFile "data/train.csv"

  -- float型の気温のみのリスト
  let train_tempature_list = case decodeByName train  of
        Left err -> []
        Right (_, v) -> make_floatlist v
  -- 7日間の気温のリストのリスト
  let train_sevendays_tempature_list = make_7days_tempature_list train_tempature_list
  print train_sevendays_tempature_list
  -- valid <- readFile("data/valid.csv")

  -- readFromFile ("data/train.csv")
  -- eval <- readFile("data/eval.csv")
  init <- sample $ LinearSpec {in_features = numFeatures, out_features = 1} -- sample parametrised classという型クラス　linear specモデルの設定を定義しているデータ型 
  -- infeatures 入ってくる次元を指定　
  randGen <- defaultRNG
  printParams init
  (trained, _) <- foldLoop (init, randGen) numIters $ \(state, randGen) i -> do
    let (input, randGen') = randn' [batchSize, numFeatures] randGen
        (y, y') = (groundTruth input, model state input)
        loss = mseLoss y y'
    when (i `mod` 100 == 0) $ do
      putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show loss
    (newParam, _) <- runStep state optimizer loss 5e-3
    pure (newParam, randGen')
  printParams trained
  pure ()
  where
    optimizer = GD
    defaultRNG = mkGenerator (Device CPU 0) 31415
    batchSize = 4
    numIters = 2000
    numFeatures = 3