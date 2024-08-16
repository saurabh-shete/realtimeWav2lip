import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import io from 'socket.io-client';
import WelcomePage from './WelcomePage';
import SecondPage from './SecondPage';
import ThirdPage from './ThirdPage';

function App() {
  const [page,setPage]=useState(1)

  useEffect(()=>{
    setTimeout(()=>{setPage(2)},2000)
  },[])

  const nextPage=()=>{
    setPage(3)
  }
  
  return (
    <div className="h-screen bg-gray-100">
     {page==1&&<WelcomePage/>}
     {page==2&&<SecondPage nextPage={nextPage} />}
     {page==3&&<ThirdPage/>}
    </div>
  );
}

export default App;
