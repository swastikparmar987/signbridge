import * as React from "react"
import { MenuBar } from "@/components/ui/bottom-menu"

const menuItems = [
  {
    icon: (props: React.SVGProps<SVGSVGElement>) => (
      <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 18 18" {...props}>
        <title>msg</title>
        <g fill="none" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" stroke="currentColor">
          <path d="M9,1.75C4.996,1.75,1.75,4.996,1.75,9c0,1.319,.358,2.552,.973,3.617,.43,.806-.053,2.712-.973,3.633,1.25,.068,2.897-.497,3.633-.973,.489,.282,1.264,.656,2.279,.848,.433,.082,.881,.125,1.338,.125,4.004,0,7.25-3.246,7.25-7.25S13.004,1.75,9,1.75Z" />
        </g>
      </svg>
    ),
    label: "Messages"
  },
  {
    icon: (props: React.SVGProps<SVGSVGElement>) => (
      <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 18 18" {...props}>
        <title>envelope</title>
        <g fill="none" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" stroke="currentColor">
          <path d="M1.75,5.75l6.767,3.733c.301,.166,.665,.166,.966,0l6.767-3.733" />
          <rect x="1.75" y="3.25" width="14.5" height="11.5" rx="2" ry="2" transform="translate(18 18) rotate(180)" />
        </g>
      </svg>
    ),
    label: "Mail"
  },
  {
    icon: (props: React.SVGProps<SVGSVGElement>) => (
      <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 18 18" {...props}>
        <title>hashtag</title>
        <g fill="none" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" stroke="currentColor">
          <line x1="3.75" y1="6.25" x2="15.25" y2="6.25" />
          <line x1="2.75" y1="11.75" x2="14.25" y2="11.75" />
          <line x1="7.633" y1="2.75" x2="5.289" y2="15.25" />
          <line x1="12.711" y1="2.75" x2="10.367" y2="15.25" />
        </g>
      </svg>
    ),
    label: "Explore"
  },
  {
    icon: (props: React.SVGProps<SVGSVGElement>) => (
      <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 18 18" {...props}>
        <title>upload-4</title>
        <g fill="none" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" stroke="currentColor">
          <path d="M15.25,11.75v1.5c0,1.105-.895,2-2,2H4.75c-1.105,0-2-.895-2-2v-1.5" />
          <polyline points="12.5 6.25 9 2.75 5.5 6.25" />
          <line x1="9" y1="2.75" x2="9" y2="10.25" />
        </g>
      </svg>
    ),
    label: "Share"
  },
  {
    icon: (props: React.SVGProps<SVGSVGElement>) => (
      <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 18 18" {...props}>
        <title>feather-plus</title>
        <g fill="none" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" stroke="currentColor">
          <path d="M13.974,9.731c-.474,3.691-3.724,4.113-6.974,3.519" />
          <path d="M3.75,16.25S5.062,4.729,16.25,3.75c-.56,.976-.573,2.605-.946,4.239-.524,2.011-2.335,2.261-4.554,2.261" />
          <line x1="4.25" y1="1.75" x2="4.25" y2="6.75" />
          <line x1="6.75" y1="4.25" x2="1.75" y2="4.25" />
        </g>
      </svg>
    ),
    label: "Write"
  },
  {
    icon: (props: React.SVGProps<SVGSVGElement>) => (
      <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 18 18" {...props}>
        <title>menu</title>
        <g fill="none" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" stroke="currentColor">
          <line x1="2.25" y1="9" x2="15.75" y2="9" />
          <line x1="2.25" y1="3.75" x2="15.75" y2="3.75" />
          <line x1="2.25" y1="14.25" x2="15.75" y2="14.25" />
        </g>
      </svg>
    ),
    label: "Menu"
  }
]

function MenuBarDemo() {
  return (
    <div className="flex items-center justify-center p-6">
      <MenuBar items={menuItems} />
    </div>
  )
}

export { MenuBarDemo }
