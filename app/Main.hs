{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}

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

data MLPSpec = MLPSpec
  { feature_counts :: [Int],
    nonlinearitySpec :: Tensor -> Tensor
  }

data MLP = MLP
  { layers :: [Linear],
    nonlinearity :: Tensor -> Tensor
  }
  deriving (Generic, Parameterized)

stringSexToFloat :: String -> Float
stringSexToFloat s =
  if s == "male" then 1.0
  else 0.0

stringEmbarkedToFloat :: String -> Float
stringEmbarkedToFloat e =
  if e=="Q" then 0.0
  else if e=="S" then 1.0
  else 2.0

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
  }

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

instance Randomizable MLPSpec MLP where
  sample MLPSpec {..} = do
    let layer_sizes = mkLayerSizes feature_counts
    linears <- mapM sample $ map (uncurry LinearSpec) layer_sizes
    return $ MLP {layers = linears, nonlinearity = nonlinearitySpec}
    where
      mkLayerSizes (a : (b : t)) =
        scanl shift (a, b) t
        where
          shift (a, b) c = (b, c)

mlp :: MLP -> Tensor -> Tensor
mlp MLP {..} input = foldl' revApply input $ intersperse nonlinearity $ map linear layers
  where
    revApply x f = f x

--------------------------------------------------------------------------------
-- Training code
-------------------------3-------------------------------------------------------

batchSize = 10

numIters = 2000
 
model :: MLP -> Tensor -> Tensor
model params t = mlp params t

main :: IO ()
main = do
  train <- BL.readFile "app/xor-mlp/data/titanic/train.csv"
  let train_titanic_list = case decodeByName train  of
        Left err -> []
        Right (_, v) -> makeNewTitanicList v

  let input_list = newTitanicToPairList train_titanic_list
      perEpoch = Prelude.length input_list `Prelude.div` batchSize
  print(Prelude.take 5 train_titanic_list)
  init <-
    sample $
      MLPSpec
        { feature_counts = [7, 7, 1],
          nonlinearitySpec = Torch.tanh
        } -- input 2 hidden 2 output 1
  (trained,losses_list) <- foldLoop (init,[]) epoch $ \(state, losses_list) i -> do
    (trained2,loss,pair_list) <- foldLoop (state,0,input_list) perEpoch $ \(state2,loss,pair_list) j -> do
    -- input <- randIO' [batchSize, 2] >>= return . (toDType Float) . (gt 0.5) -- 2*2 (>.) = gt 0.5より大きいか小さいかで
    
      let start_number = j * batchSize
          end_number = start_number + batchSize
          (y_float,input_float) = unzip (Prelude.take (end_number - start_number) (drop start_number pair_list))
          input = asTensor input_float
          (y, y') = (asTensor y_float, squeezeAll $ model state input) --　逆伝播
          newloss = mseLoss y y' 
          -- new_shuffle_list <- System.Random.Shuffle.shuffleM pair_list
      
      (newState, _) <- runStep state2 optimizer loss 1e-6 
      
      return (newState,newloss,pair_list)
      
      -- when (i `mod` 100 == 0) $ do
      --   putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show loss
    return (trained2,[loss]++losses_list)
  -- putStrLn "Final Model:"
  -- putStrLn $ "0, 0 => " ++ (show $ squeezeAll $ model trained (asTensor [0, 0 :: Float])) -- xor 0
  -- putStrLn $ "0, 1 => " ++ (show $ squeezeAll $ model trained (asTensor [0, 1 :: Float])) -- xor 1
  -- putStrLn $ "1, 0 => " ++ (show $ squeezeAll $ model trained (asTensor [1, 0 :: Float])) -- xor 1
  -- putStrLn $ "1, 1 => " ++ (show $ squeezeAll $ model trained (asTensor [1, 1 :: Float])) -- xor 0
  drawLearningCurve "/home/acf16407il/.local/lib/hasktorch/bordeaux-intern2024/app/xor-mlp/images/loss.png" "Learning Curve" [("train_loss",map asValue (reverse losses_list)),("valid_loss",map asValue (reverse losses_list))] 
  pure ()
  return ()
  where
    optimizer = GD
    epoch = 1000
  
    -- tensorXOR t = (1 - (1 - a) * (1 - b)) * (1 - (a * b)) -- 1-a:not a a*b:and　と考えると (not(not a and not b)and(not(a and b))
    --   where
    --     a = select 1 0 t --  0番目をとってくる
    --     b = select 1 1 t --  1番目をとってくる