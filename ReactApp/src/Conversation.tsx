import React, { useEffect, useState } from 'react';
import { Input, Button, Upload, message, Spin, Typography, Checkbox, Divider } from 'antd';
import type { UploadProps } from 'antd';
import { UploadOutlined, LoadingOutlined } from '@ant-design/icons';
import ReactMarkdown, { Components } from 'react-markdown';
import './css/Conversation.css';
import Conversation_Handler, { ConversationHeaders } from './Api_Handlers/Conversation_Handler';
import { ServerResponse } from './Api_Handlers/Fetch_Helper';

import { Image } from 'antd';
import { List } from 'antd';
import Title from 'antd/es/typography/Title';

interface Message {
    id: string;
    text: string;
    sender: string;
}

interface ConversationProps {
    conversationId: string;
}

const antd_components = {
      // Headers
  h1: ({ children }) => <Typography.Title level={1}>{children}</Typography.Title>,
  h2: ({ children }) => <Typography.Title level={2}>{children}</Typography.Title>,
  h3: ({ children }) => <Typography.Title level={3}>{children}</Typography.Title>,
  h4: ({ children }) => <Typography.Title level={4}>{children}</Typography.Title>,
  h5: ({ children }) => <Typography.Title level={5}>{children}</Typography.Title>,
  h6: 'h5', // Not supported, so use h5 instead
  // Paragraph
  p: ({ children }) => <Typography.Paragraph>{children}</Typography.Paragraph>,

  // Emphasis
  em: ({ children }) => <Typography.Text italic>{children}</Typography.Text>,

  // Strong
  strong: ({ children }) => <Typography.Text strong>{children}</Typography.Text>,

  // Blockquote
  blockquote: ({ children }) => (
    <Typography.Paragraph italic style={{ padding: '10px 20px', backgroundColor: '#f5f5f5', borderLeft: '5px solid #ccc' }}>
      {children}
    </Typography.Paragraph>
  ),

  // Links
  a: ({ children, href }) => (
    <Typography.Link href={href} target="_blank" rel="noopener noreferrer">
      {children}
    </Typography.Link>
  ),

  // Lists
  ul: ({ children }) => <List>{children}</List>,
  ol: ({ children }) => <List.Item>{children}</List.Item>,
  li: ({ children }) => <List.Item>{children}</List.Item>,

  // Checkboxes (inside lists)
  input: ({ checked, type }) =>
    type === 'checkbox' ? <Checkbox checked={checked} disabled /> : null,

  // Images
  img: ({ alt, src }) => <Image alt={alt} src={src} style={{ maxWidth: '100%' }} />,

  // Horizontal Rule
  hr: () => <Divider />,

  // Code blocks
  code: ({ children }) => {
    return (
      <Typography.Paragraph code style={{ backgroundColor: '#f5f5f5', padding: '10px', borderRadius: '4px' }}>
        {children}
      </Typography.Paragraph>
    );
},
} as Partial<Components>;

const Conversation: React.FC<ConversationProps> = ({ conversationId }) => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [conversationHeaders, setConversationHeaders] = useState<ConversationHeaders>();
    const [inputValue, setInputValue] = useState<string>('');
    const [conversationAttachments, setConversationAttachments] = useState<string[]>([]);
    const [messagesLoaded, setMessagesLoaded] = useState<boolean>(false);
    const [isSending, setIsSending] = useState<boolean>(false);

    const props: UploadProps = {
        name: 'file',
        action: import.meta.env.VITE_BACKEND_URL + '/conversationAttachment/' + conversationId,
        onChange(info) {
            if (info.file.status !== 'uploading') {
                console.log(info.file, info.fileList);
            }
            if (info.file.status === 'done') {
                message.success(`${info.file.name} file uploaded successfully`);
                console.log(info.file.xhr.response);
            } else if (info.file.status === 'error') {
                message.error(`${info.file.name} file upload failed.`);
            }
        },
    };

    useEffect(() => {
        if (conversationId) {
            loadConversation(conversationId);
        }
    }, [conversationId]);

    const loadConversation = (id: string) => {
        setMessagesLoaded(false); // Show loader while loading
        Conversation_Handler.loadFullConversation({ id: id, title: '', createdAt: '' }).then((response) => {
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
        setIsSending(true); // Show loader while sending message

        Conversation_Handler.sendMessage({ id: conversationId, title: '', createdAt: '' }, newMessage).then((response) => {
            setMessages(response.messages);
            setConversationHeaders(response.header);
            //setConversationAttachments(response.attachments.map((attachment) => attachment.url));
        }).catch((error: ServerResponse) => {
            console.log(error);
            message.error(error.message);
        }).finally(() => {
            setIsSending(false); // Hide loader after message is sent
        });
    };

    // We need a function that listens for user key presses and sends the message when the user presses Enter (And Shift+Enter for new line)
    // This function will be called whenever the user presses a key
    const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); // Prevent new line
            handleSendMessage(); // Send the message
        } 
    };
    // Add the onKeyPress event listener to the Input.TextArea component



    function removeMarkdownCodeBlocks(text: string): string {
        // Regular expression to match ```markdown ... ``` code blocks
        const markdownCodeBlockRegex = /```markdown([\s\S]*?)```/g;
        
        // Replace all matches with an empty string
        const cleanedText = text.replace(markdownCodeBlockRegex, '$1');
        
        return cleanedText;
    }
    const renderMessageContent = (message: Message) => {
        return (
            <ReactMarkdown components={antd_components} children={removeMarkdownCodeBlocks(message.text)} />
        );
    };


    return (
        <div className='conversation_container'>
           
            <Title>{conversationHeaders?.title}</Title>
            <List
                className='message_list'
                dataSource={messages}
                loading={isSending}
                renderItem={(message) => (
                    <List.Item>
                        <div
                            style={{
                                textAlign: message.sender === 'user' ? 'right' : 'left',
                                width: '100%',
                            }}
                        >
                            <strong>{message.sender === 'user' ? 'You' : 'Bot'}:</strong> 
                            {
                                renderMessageContent(message)
                            }
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
                    onPressEnter={handleKeyPress}
                />
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Upload {...props}>
                        <Button icon={<UploadOutlined />}>Upload File</Button>
                    </Upload>
                    <Button type="primary" onClick={handleSendMessage} disabled={isSending}>
                        {isSending ? <Spin indicator={<LoadingOutlined style={{ color: 'white' }} spin />} /> : 'Send'}
                    </Button>
                </div>
            </div>
        </div>
    );
};

export default Conversation;
