import React, { useState } from 'react';
import axios from 'axios';


const SecondPage = ({ nextPage }) => {
    const [image, setImage] = useState(null);
    const [imageUrl, setImageUrl] = useState(null);

    const handleImageChange = (event) => {
        setImage(event.target.files[0]);
    };

    const uploadImage = async () => {
        if (!image) return console.log('Please select an image to upload');

        const formData = new FormData();
        formData.append('image', image);
        try {
            const response = await axios.post('http://127.0.0.1:5000/upload/image', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            setImageUrl(response.data.url);
        } catch (error) {
            console.error('Error uploading image:', error);
        }
    };


    return (
        <div className="bg-white p-8 flex flex-col items-center h-full">
            <h1 className="text-3xl font-bold mb-4 text-center">Avatar Generation</h1>
            <input
                type="file"
                accept="image/*"
                onChange={handleImageChange}
                className="w-1/3 text-sm text-gray-900 border border-gray-300 cursor-pointer bg-gray-50 focus:outline-none focus:border-indigo-500"
            />
            {imageUrl && (
                <div className="mt-4 max-h-[50vh] flex justify-center">
                    <img
                        src={imageUrl}
                        alt="Uploaded"
                        className="object-contain rounded-lg shadow"
                    // style={{ maxHeight: '300px', maxWidth: '300px' }}
                    />
                </div>
            )}
            <div className='w-1/2 flex justify-between gap-5 items-center'>
                <button
                    onClick={uploadImage}
                    className="mt-4 w-full bg-indigo-500 text-white font-semibold py-2 px-4 rounded-lg hover:bg-indigo-600 focus:outline-none focus:ring-2 focus:ring-indigo-400"
                >
                    Upload Image
                </button>
                <button
                    onClick={nextPage}
                    className="mt-4 w-full bg-green-500 text-white font-semibold py-2 px-4 rounded-lg hover:bg-green-600 focus:outline-none focus:ring-2 focus:ring-indigo-400"
                >
                    Generate Avatar
                </button>
            </div>
      
        </div>
    )
}

export default SecondPage