import React, { useState } from 'react';
import { Button, Card, CardContent } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Speaker } from 'lucide-react';

const TextToSpeechApp = () => {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);

  const handleTextChange = (e) => {
    setText(e.target.value);
  };

  const convertTextToSpeech = async () => {
    if (!text.trim()) return;
    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/api/text-to-speech', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });
      if (!response.ok) {
        throw new Error('Failed to convert text to speech');
      }
      const blob = await response.blob();
      const audioURL = window.URL.createObjectURL(blob);
      const audio = new Audio(audioURL);
      audio.play();
    } catch (error) {
      alert(error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
      <Card className="w-full max-w-2xl p-4 shadow-md">
        <CardContent>
          <h1 className="text-2xl font-bold mb-4">Text-to-Speech Converter</h1>
          <Textarea
            value={text}
            onChange={handleTextChange}
            placeholder="Enter text here..."
            className="w-full h-32 p-2 border rounded mb-4"
          />
          <Button onClick={convertTextToSpeech} disabled={loading} className="flex items-center gap-2">
            <Speaker />
            {loading ? 'Converting...' : 'Convert to Voice'}
          </Button>
        </CardContent>
      </Card>
    </div>
  );
};

export default TextToSpeechApp;