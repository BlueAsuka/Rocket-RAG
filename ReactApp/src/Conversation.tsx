import React, { useEffect, useState } from 'react';
import { Input, Button, Upload, List, message } from 'antd';
import { UploadOutlined } from '@ant-design/icons';
import './css/Conversation.css';
import Conversation_Handler, { ConversationHeaders } from './Api_Handlers/Conversation_Handler';
import { ServerResponse } from './Api_Handlers/Fetch_Helper';

interface Message {
    id: string;
    text: string;
    sender: string; // Change the type of 'sender' to string
}

interface ConversationProps {
    conversationId: string;
  }

const Conversation: React.FC<ConversationProps> = ({conversationId}) => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [conversationHeaders, setConversationHeaders] = useState<ConversationHeaders>();
    const [inputValue, setInputValue] = useState<string>('');
    const [conversationAttachments, setConversationAttachments] = useState<string[]>([]);
    const [messagesLoaded, setMessagesLoaded] = useState<boolean>(false);

    useEffect(() => {
        if (conversationId) { // Prevent undefined conversationId
            loadConversation(conversationId);
        }
    }, [conversationId]);

    const loadConversation = (id: string) => {

        Conversation_Handler.loadFullConversation({id: id, title: '', createdAt: ''}).then((response) => {
            setMessages(response.messages);
            setConversationHeaders(response.header);
            setMessagesLoaded(true);
        }).catch((error: ServerResponse) => {
            console.log(error);
            message.error(error.message);
        });
    };

    const handleSendMessage = () => {
        if (!inputValue.trim()) return;

        const newMessage: Message = {
            id: `${messages.length + 1}`,
            text: inputValue,
            sender: 'user',
        };

        setMessages([...messages, newMessage]);
        setInputValue('');

        Conversation_Handler.sendMessage({id: conversationId, title: '', createdAt: ''}, newMessage).then((response) => {
            console.log(response);
            setMessages(response.messages);
            setConversationHeaders(response.header);
            setConversationAttachments(response.attachments.map((attachment) => attachment.url));
        }).catch((error: ServerResponse) => {
            console.log(error);
            message.error(error.message);
        }
        );
    };

    const handleFileUpload = () => {
        message.error('File upload not implemented yet');
        console.log(conversationAttachments);
        // Implement the file upload logic here
    };

    return (
        <div className='conversation_container'>
            <h1> {conversationHeaders?.title} </h1>
            <List
                className='message_list'
                dataSource={messages}
                loading={!messagesLoaded}
                renderItem={(message) => (
                    <List.Item>
                        <div
                            style={{
                                textAlign: message.sender === 'user' ? 'right' : 'left',
                                width: '100%',
                            }}
                        >
                            <strong>{message.sender === 'user' ? 'You' : 'Bot'}:</strong> {message.text}
                        </div>
                    </List.Item>
                )}
            />
            <div className='input_conversation'>
                <Input.TextArea
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    autoSize={{ minRows: 3, maxRows: 5 }}

                    placeholder="Type your message..."
                    style={{ marginBottom: 10 }}
                />

                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    {/* Before upload function helps with passing the file before handleFileUpload function */}
                    <Upload showUploadList={false} beforeUpload={() => false} onChange={handleFileUpload}>
                        <Button icon={<UploadOutlined />}>Upload File</Button>
                    </Upload>

                    <Button type="primary" onClick={handleSendMessage}>
                        Send
                    </Button>
                </div>
            </div>
        </div>
    );
};

export default Conversation;
