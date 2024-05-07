{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE OverloadedRecordDot #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE BangPatterns      #-}
{-# LANGUAGE DeriveGeneric     #-}
{-# LANGUAGE LambdaCase        #-}
{-# LANGUAGE OverloadedStrings #-}

module Main where

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
import Data.Csv.Incremental
import Data.Csv (FromRecord, ToRecord)
import Data.Text (Text)

data Tempature = Tempature {
  date :: !Text,
  dairy_mean_tempature :: !Float }
  deriving (Generic,Show)

instance FromRecord Tempature
instance ToRecord Tempature

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
feed :: (ByteString -> Parser Tempature) -> Handle -> IO (Parser Tempature)
feed k csvFile = do
  hIsEOF csvFile >>= \case
    True  -> return $ k empty
    False -> k <$> hGetSome csvFile 4096

readFromFile :: IO ()
readFromFile = do
  withFile "data/valid.csv" ReadMode $ \ csvFile -> do
    let loop !_ (Fail _ errMsg) = do putStrLn errMsg; exitFailure
        loop acc (Many rs k)    = loop (acc <> rs) =<< feed k csvFile
        loop acc (Done rs)      = print (acc <> rs)
    loop [] (decode NoHeader)

main :: IO ()
main = do
  readFromFile
  init <- sample $ LinearSpec {in_features = numFeatures, out_features = 1}
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